// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "FilterNode.h"

namespace milvus {
namespace exec {
PhyFilterNode::PhyFilterNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::FilterNode>& filter)
    : Operator(driverctx,
               filter->output_type(),
               operator_id,
               filter->id(),
               "PhyFilterNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    std::vector<expr::TypedExprPtr> filters;
    filters.emplace_back(filter->filter());
    exprs_ = std::make_unique<ExprSet>(filters, exec_context);
    need_process_rows_ = query_context_->get_active_count();
    num_processed_rows_ = 0;
}

void
PhyFilterNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

bool
PhyFilterNode::IsFinished() {
    return is_finished_;
}

inline size_t
find_binsert_position(const std::vector<float>& distances,
                      size_t lo,
                      size_t hi,
                      float dist) {
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (distances[mid] > dist) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

RowVectorPtr
PhyFilterNode::GetOutput() {
    if (is_finished_ || !no_more_input_) {
        return nullptr;
    }

    DeferLambda([&]() { is_finished_ = true; });

    if (input_ == nullptr) {
        return nullptr;
    }

    std::chrono::high_resolution_clock::time_point scalar_start =
        std::chrono::high_resolution_clock::now();

    milvus::SearchResult search_result = query_context_->get_search_result();
    if (search_result.vector_iterators_.has_value()) {
        AssertInfo(search_result.vector_iterators_.value().size() ==
                       search_result.total_nq_,
                   "Vector Iterators' count must be equal to total_nq_, Check "
                   "your code");
        int nq_index = 0;
        int64_t nq = search_result.total_nq_;
        int64_t unity_topk = search_result.unity_topK_;
        AssertInfo(nq = search_result.vector_iterators_.value().size(),
                   "nq and iterator not equal size");
        // LOG_INFO("nq {}, unity topk {}, size {}",
        //          nq_index,
        //          unity_topk,
        //          search_result.vector_iterators_.value().size());
        search_result.seg_offsets_.resize(nq * unity_topk);
        search_result.distances_.resize(nq * unity_topk);
        for (auto& iterator : search_result.vector_iterators_.value()) {
            EvalCtx eval_ctx(operator_context_->get_exec_context(),
                             exprs_.get());
            int topk = 0;
            while (iterator->HasNext() && topk < unity_topk) {
                FixedVector<int64_t> offsets, diss;
                offsets.reserve(unity_topk);
                diss.reserve(unity_topk);
                while (iterator->HasNext()) {
                    auto offset_dis_pair = iterator->Next();
                    AssertInfo(
                        offset_dis_pair.has_value(),
                        "Wrong state! iterator cannot return valid result "
                        "whereas it still"
                        "tells hasNext, terminate operation");
                    auto offset = offset_dis_pair.value().first;
                    auto dis = offset_dis_pair.value().second;
                    offsets.emplace_back(offset);
                    diss.emplace_back(dis);
                    if (offsets.size() == unity_topk) {
                        break;
                    }
                }
                // for (int j = 0; j < offsets.size(); ++j) {
                //     LOG_INFO("offset {}: {}", j, offsets[j]);
                // }
                // auto x = std::make_shared<ColumnVector>(std::move(offsets));
                eval_ctx.set_input(&offsets);
                exprs_->Eval(0, 1, true, eval_ctx, results_);
                AssertInfo(
                    results_.size() == 1 && results_[0] != nullptr,
                    "PhyFilterNode result size should be size one and not "
                    "be nullptr");

                auto col_vec =
                    std::dynamic_pointer_cast<ColumnVector>(results_[0]);
                auto col_vec_size = col_vec->size();
                TargetBitmapView bitsetview(col_vec->GetRawData(),
                                            col_vec_size);
                Assert(bitsetview.size() <= unity_topk);
                for (auto i = 0; i < bitsetview.size(); ++i) {
                    if (bitsetview[i] > 0) {
                        auto pos =
                            find_binsert_position(search_result.distances_,
                                                  nq_index * unity_topk,
                                                  nq_index * unity_topk + topk,
                                                  diss[i]);
                        if (topk > pos) {
                            std::memmove(&search_result.distances_[pos + 1],
                                         &search_result.distances_[pos],
                                         (topk - pos) * sizeof(float));
                            std::memmove(&search_result.seg_offsets_[pos + 1],
                                         &search_result.seg_offsets_[pos],
                                         (topk - pos) * sizeof(int64_t));
                        }
                        search_result.seg_offsets_[pos] = offsets[i];
                        search_result.distances_[pos] = diss[i];
                        ++topk;
                        if (topk == unity_topk) {
                            break;
                        }
                    }
                }
                if (topk == unity_topk) {
                    break;
                }
            }
            nq_index++;
        }
    }
    query_context_->set_search_result(std::move(search_result));
    std::chrono::high_resolution_clock::time_point scalar_end =
        std::chrono::high_resolution_clock::now();
    double scalar_cost =
        std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
            .count();
    LOG_INFO("FUCK post filter cost: {}", scalar_cost);
    monitor::internal_core_search_latency_postfilter.Observe(scalar_cost);

    return input_;
}

}  // namespace exec
}  // namespace milvus
