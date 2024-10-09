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

#include "CompareExpr.h"
#include "common/type_c.h"
#include <optional>
#include "query/Relational.h"

namespace milvus {
namespace exec {

bool
PhyCompareFilterExpr::IsStringExpr() {
    return expr_->left_data_type_ == DataType::VARCHAR ||
           expr_->right_data_type_ == DataType::VARCHAR;
}

int64_t
PhyCompareFilterExpr::GetNextBatchSize() {
    auto current_rows = GetCurrentRows();

    return current_rows + batch_size_ >= active_count_
               ? active_count_ - current_rows
               : batch_size_;
}

template <typename T>
MultipleChunkDataAccessor
PhyCompareFilterExpr::GetChunkData(FieldId field_id,
                                   bool index,
                                   int64_t& current_chunk_id,
                                   int64_t& current_chunk_pos) {
    if (index) {
        auto& indexing = const_cast<index::ScalarIndex<T>&>(
            segment_->chunk_scalar_index<T>(field_id, current_chunk_id));
        auto current_chunk_size = segment_->type() == SegmentType::Growing
                                      ? size_per_chunk_
                                      : active_count_;

        if (indexing.HasRawData()) {
            return [&, current_chunk_size]() -> const number {
                if (current_chunk_pos >= current_chunk_size) {
                    current_chunk_id++;
                    current_chunk_pos = 0;
                    indexing = const_cast<index::ScalarIndex<T>&>(
                        segment_->chunk_scalar_index<T>(field_id,
                                                        current_chunk_id));
                }
                auto raw = indexing.Reverse_Lookup(current_chunk_pos);
                current_chunk_pos++;
                if (!raw.has_value()) {
                    return std::nullopt;
                }
                return raw.value();
            };
        }
    }
    auto chunk_data =
        segment_->chunk_data<T>(field_id, current_chunk_id).data();
    auto chunk_valid_data =
        segment_->chunk_data<T>(field_id, current_chunk_id).valid_data();
    auto current_chunk_size = segment_->chunk_size(field_id, current_chunk_id);
    return
        [=, &current_chunk_id, &current_chunk_pos]() mutable -> const number {
            if (current_chunk_pos >= current_chunk_size) {
                current_chunk_id++;
                current_chunk_pos = 0;
                chunk_data =
                    segment_->chunk_data<T>(field_id, current_chunk_id).data();
                chunk_valid_data =
                    segment_->chunk_data<T>(field_id, current_chunk_id)
                        .valid_data();
                current_chunk_size =
                    segment_->chunk_size(field_id, current_chunk_id);
            }
            if (chunk_valid_data && !chunk_valid_data[current_chunk_pos]) {
                current_chunk_pos++;
                return std::nullopt;
            }
            return chunk_data[current_chunk_pos++];
        };
}

template <>
MultipleChunkDataAccessor
PhyCompareFilterExpr::GetChunkData<std::string>(FieldId field_id,
                                                bool index,
                                                int64_t& current_chunk_id,
                                                int64_t& current_chunk_pos) {
    if (index) {
        auto& indexing = const_cast<index::ScalarIndex<std::string>&>(
            segment_->chunk_scalar_index<std::string>(field_id,
                                                      current_chunk_id));
        auto current_chunk_size = segment_->type() == SegmentType::Growing
                                      ? size_per_chunk_
                                      : active_count_;

        if (indexing.HasRawData()) {
            return [&, current_chunk_size]() mutable -> const number {
                if (current_chunk_pos >= current_chunk_size) {
                    current_chunk_id++;
                    current_chunk_pos = 0;
                    indexing = const_cast<index::ScalarIndex<std::string>&>(
                        segment_->chunk_scalar_index<std::string>(
                            field_id, current_chunk_id));
                }
                auto raw = indexing.Reverse_Lookup(current_chunk_pos);
                current_chunk_pos++;
                if (!raw.has_value()) {
                    return std::nullopt;
                }
                return raw.value();
            };
        }
    }
    if (segment_->type() == SegmentType::Growing &&
        !storage::MmapManager::GetInstance()
             .GetMmapConfig()
             .growing_enable_mmap) {
        auto chunk_data =
            segment_->chunk_data<std::string>(field_id, current_chunk_id)
                .data();
        auto chunk_valid_data =
            segment_->chunk_data<std::string>(field_id, current_chunk_id)
                .valid_data();
        auto current_chunk_size =
            segment_->chunk_size(field_id, current_chunk_id);
        return [=,
                &current_chunk_id,
                &current_chunk_pos]() mutable -> const number {
            if (current_chunk_pos >= current_chunk_size) {
                current_chunk_id++;
                current_chunk_pos = 0;
                chunk_data =
                    segment_
                        ->chunk_data<std::string>(field_id, current_chunk_id)
                        .data();
                chunk_valid_data =
                    segment_
                        ->chunk_data<std::string>(field_id, current_chunk_id)
                        .valid_data();
                current_chunk_size =
                    segment_->chunk_size(field_id, current_chunk_id);
            }
            if (chunk_valid_data && !chunk_valid_data[current_chunk_pos]) {
                current_chunk_pos++;
                return std::nullopt;
            }
            return chunk_data[current_chunk_pos++];
        };
    } else {
        auto chunk_data =
            segment_->chunk_view<std::string_view>(field_id, current_chunk_id)
                .first.data();
        auto chunk_valid_data =
            segment_->chunk_data<std::string_view>(field_id, current_chunk_id)
                .valid_data();
        auto current_chunk_size =
            segment_->chunk_size(field_id, current_chunk_id);
        return [=,
                &current_chunk_id,
                &current_chunk_pos]() mutable -> const number {
            if (current_chunk_pos >= current_chunk_size) {
                current_chunk_id++;
                current_chunk_pos = 0;
                chunk_data = segment_
                                 ->chunk_view<std::string_view>(
                                     field_id, current_chunk_id)
                                 .first.data();
                chunk_valid_data = segment_
                                       ->chunk_data<std::string_view>(
                                           field_id, current_chunk_id)
                                       .valid_data();
                current_chunk_size =
                    segment_->chunk_size(field_id, current_chunk_id);
            }
            if (chunk_valid_data && !chunk_valid_data[current_chunk_pos]) {
                current_chunk_pos++;
                return std::nullopt;
            }

            return std::string(chunk_data[current_chunk_pos++]);
        };
    }
}

MultipleChunkDataAccessor
PhyCompareFilterExpr::GetChunkData(DataType data_type,
                                   FieldId field_id,
                                   bool index,
                                   int64_t& current_chunk_id,
                                   int64_t& current_chunk_pos) {
    switch (data_type) {
        case DataType::BOOL:
            return GetChunkData<bool>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::INT8:
            return GetChunkData<int8_t>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::INT16:
            return GetChunkData<int16_t>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::INT32:
            return GetChunkData<int32_t>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::INT64:
            return GetChunkData<int64_t>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::FLOAT:
            return GetChunkData<float>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::DOUBLE:
            return GetChunkData<double>(
                field_id, index, current_chunk_id, current_chunk_pos);
        case DataType::VARCHAR: {
            return GetChunkData<std::string>(
                field_id, index, current_chunk_id, current_chunk_pos);
        }
        default:
            PanicInfo(DataTypeInvalid, "unsupported data type: {}", data_type);
    }
}

template <typename OpType>
VectorPtr
PhyCompareFilterExpr::ExecCompareExprDispatcher(OpType op,
                                                OffsetVector* input) {
    // take offsets as input
    if (has_input_) {
        auto real_batch_size = input->size();
        if (real_batch_size == 0) {
            return nullptr;
        }

        auto res_vec = std::make_shared<ColumnVector>(
            TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
        TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
        TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
        valid_res.set();

        auto left_data_barrier =
            segment_->num_chunk_data(expr_->left_field_id_);
        auto right_data_barrier =
            segment_->num_chunk_data(expr_->right_field_id_);

        int64_t processed_rows = 0;
        for (auto i = 0; i < real_batch_size; ++i) {
            auto offset = (*input)[i];
            auto [chunk_id,
                  chunk_offset] = [&]() -> std::pair<int64_t, int64_t> {
                if (segment_->type() == SegmentType::Growing) {
                    return {offset / size_per_chunk_, offset % size_per_chunk_};
                } else if (segment_->is_chunked()) {
                    return segment_->get_chunk_by_offset(left_field_, offset);
                } else {
                    return {0, offset};
                }
            }();
            auto left = GetChunkData(expr_->left_data_type_,
                                     expr_->left_field_id_,
                                     chunk_id,
                                     left_data_barrier);
            auto right = GetChunkData(expr_->right_data_type_,
                                      expr_->right_field_id_,
                                      chunk_id,
                                      right_data_barrier);
            auto left_opt = left(chunk_offset);
            auto right_opt = right(chunk_offset);
            if (!left_opt.has_value() || !right_opt.has_value()) {
                res[processed_rows] = false;
                valid_res[processed_rows] = false;
            } else {
                res[processed_rows] = boost::apply_visitor(
                    milvus::query::Relational<decltype(op)>{},
                    left_opt.value(),
                    right_opt.value());
            }
            processed_rows++;
        }
        return res_vec;
    }

    // normal path
    if (segment_->is_chunked()) {
        auto real_batch_size = GetNextBatchSize();
        if (real_batch_size == 0) {
            return nullptr;
        }

        auto res_vec = std::make_shared<ColumnVector>(
            TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
        TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
        TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
        valid_res.set();

        auto left = GetChunkData(expr_->left_data_type_,
                                 expr_->left_field_id_,
                                 is_left_indexed_,
                                 left_current_chunk_id_,
                                 left_current_chunk_pos_);
        auto right = GetChunkData(expr_->right_data_type_,
                                  expr_->right_field_id_,
                                  is_right_indexed_,
                                  right_current_chunk_id_,
                                  right_current_chunk_pos_);
        for (int i = 0; i < real_batch_size; ++i) {
            if (!left().has_value() || !right().has_value()) {
                res[i] = false;
                valid_res[i] = false;
                continue;
            }
            res[i] =
                boost::apply_visitor(milvus::query::Relational<decltype(op)>{},
                                     left().value(),
                                     right().value());
        }
        return res_vec;
    } else {
        auto real_batch_size = GetNextBatchSize();
        if (real_batch_size == 0) {
            return nullptr;
        }

        auto res_vec = std::make_shared<ColumnVector>(
            TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
        TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
        TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
        valid_res.set();

        auto left_data_barrier =
            segment_->num_chunk_data(expr_->left_field_id_);
        auto right_data_barrier =
            segment_->num_chunk_data(expr_->right_field_id_);

        int64_t processed_rows = 0;
        for (int64_t chunk_id = current_chunk_id_; chunk_id < num_chunk_;
             ++chunk_id) {
            auto chunk_size = chunk_id == num_chunk_ - 1
                                  ? active_count_ - chunk_id * size_per_chunk_
                                  : size_per_chunk_;
            auto left = GetChunkData(expr_->left_data_type_,
                                     expr_->left_field_id_,
                                     chunk_id,
                                     left_data_barrier);
            auto right = GetChunkData(expr_->right_data_type_,
                                      expr_->right_field_id_,
                                      chunk_id,
                                      right_data_barrier);

            for (int i = chunk_id == current_chunk_id_ ? current_chunk_pos_ : 0;
                 i < chunk_size;
                 ++i) {
                if (!left(i).has_value() || !right(i).has_value()) {
                    res[processed_rows] = false;
                    valid_res[processed_rows] = false;
                } else {
                    res[processed_rows] = boost::apply_visitor(
                        milvus::query::Relational<decltype(op)>{},
                        left(i).value(),
                        right(i).value());
                }
                processed_rows++;

                if (processed_rows >= batch_size_) {
                    current_chunk_id_ = chunk_id;
                    current_chunk_pos_ = i + 1;
                    return res_vec;
                }
            }
        }
        return res_vec;
    }
}

template <typename T>
ChunkDataAccessor
PhyCompareFilterExpr::GetChunkData(FieldId field_id,
                                   int chunk_id,
                                   int data_barrier) {
    if (chunk_id >= data_barrier) {
        auto& indexing = segment_->chunk_scalar_index<T>(field_id, chunk_id);
        if (indexing.HasRawData()) {
            return [&indexing](int i) -> const number {
                auto raw = indexing.Reverse_Lookup(i);
                if (!raw.has_value()) {
                    return std::nullopt;
                }
                return raw.value();
            };
        }
    }
    auto chunk_data = segment_->chunk_data<T>(field_id, chunk_id).data();
    auto chunk_valid_data =
        segment_->chunk_data<T>(field_id, chunk_id).valid_data();
    return [chunk_data, chunk_valid_data](int i) -> const number {
        if (chunk_valid_data && !chunk_valid_data[i]) {
            return std::nullopt;
        }
        return chunk_data[i];
    };
}

template <>
ChunkDataAccessor
PhyCompareFilterExpr::GetChunkData<std::string>(FieldId field_id,
                                                int chunk_id,
                                                int data_barrier) {
    if (chunk_id >= data_barrier) {
        auto& indexing =
            segment_->chunk_scalar_index<std::string>(field_id, chunk_id);
        if (indexing.HasRawData()) {
            return [&indexing](int i) -> const number {
                auto raw = indexing.Reverse_Lookup(i);
                if (!raw.has_value()) {
                    return std::nullopt;
                }
                return raw.value();
            };
        }
    }
    if (segment_->type() == SegmentType::Growing &&
        !storage::MmapManager::GetInstance()
             .GetMmapConfig()
             .growing_enable_mmap) {
        auto chunk_data =
            segment_->chunk_data<std::string>(field_id, chunk_id).data();
        auto chunk_valid_data =
            segment_->chunk_data<std::string>(field_id, chunk_id).valid_data();
        return [chunk_data, chunk_valid_data](int i) -> const number {
            if (chunk_valid_data && !chunk_valid_data[i]) {
                return std::nullopt;
            }
            return chunk_data[i];
        };
    } else {
        auto chunk_info =
            segment_->chunk_view<std::string_view>(field_id, chunk_id);
        auto chunk_data = chunk_info.first.data();
        auto chunk_valid_data = chunk_info.second.data();
        return [chunk_data, chunk_valid_data](int i) -> const number {
            if (chunk_valid_data && !chunk_valid_data[i]) {
                return std::nullopt;
            }
            return std::string(chunk_data[i]);
        };
    }
}

ChunkDataAccessor
PhyCompareFilterExpr::GetChunkData(DataType data_type,
                                   FieldId field_id,
                                   int chunk_id,
                                   int data_barrier) {
    switch (data_type) {
        case DataType::BOOL:
            return GetChunkData<bool>(field_id, chunk_id, data_barrier);
        case DataType::INT8:
            return GetChunkData<int8_t>(field_id, chunk_id, data_barrier);
        case DataType::INT16:
            return GetChunkData<int16_t>(field_id, chunk_id, data_barrier);
        case DataType::INT32:
            return GetChunkData<int32_t>(field_id, chunk_id, data_barrier);
        case DataType::INT64:
            return GetChunkData<int64_t>(field_id, chunk_id, data_barrier);
        case DataType::FLOAT:
            return GetChunkData<float>(field_id, chunk_id, data_barrier);
        case DataType::DOUBLE:
            return GetChunkData<double>(field_id, chunk_id, data_barrier);
        case DataType::VARCHAR: {
            return GetChunkData<std::string>(field_id, chunk_id, data_barrier);
        }
        default:
            PanicInfo(DataTypeInvalid, "unsupported data type: {}", data_type);
    }
}

void
PhyCompareFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    auto input = context.get_input();
    SetHasInput((input != nullptr));
    // For segment both fields has no index, can use SIMD to speed up.
    // Avoiding too much call stack that blocks SIMD.
    if (!is_left_indexed_ && !is_right_indexed_ && !IsStringExpr()) {
        result = ExecCompareExprDispatcherForBothDataSegment(input);
        return;
    }
    result = ExecCompareExprDispatcherForHybridSegment(input);
}

VectorPtr
PhyCompareFilterExpr::ExecCompareExprDispatcherForHybridSegment(
    OffsetVector* input) {
    switch (expr_->op_type_) {
        case OpType::Equal: {
            return ExecCompareExprDispatcher(std::equal_to<>{}, input);
        }
        case OpType::NotEqual: {
            return ExecCompareExprDispatcher(std::not_equal_to<>{}, input);
        }
        case OpType::GreaterEqual: {
            return ExecCompareExprDispatcher(std::greater_equal<>{}, input);
        }
        case OpType::GreaterThan: {
            return ExecCompareExprDispatcher(std::greater<>{}, input);
        }
        case OpType::LessEqual: {
            return ExecCompareExprDispatcher(std::less_equal<>{}, input);
        }
        case OpType::LessThan: {
            return ExecCompareExprDispatcher(std::less<>{}, input);
        }
        case OpType::PrefixMatch: {
            return ExecCompareExprDispatcher(
                milvus::query::MatchOp<OpType::PrefixMatch>{}, input);
        }
            // case OpType::PostfixMatch: {
            // }
        default: {
            PanicInfo(OpTypeInvalid, "unsupported optype: {}", expr_->op_type_);
        }
    }
}

VectorPtr
PhyCompareFilterExpr::ExecCompareExprDispatcherForBothDataSegment(
    OffsetVector* input) {
    switch (expr_->left_data_type_) {
        case DataType::BOOL:
            return ExecCompareLeftType<bool>(input);
        case DataType::INT8:
            return ExecCompareLeftType<int8_t>(input);
        case DataType::INT16:
            return ExecCompareLeftType<int16_t>(input);
        case DataType::INT32:
            return ExecCompareLeftType<int32_t>(input);
        case DataType::INT64:
            return ExecCompareLeftType<int64_t>(input);
        case DataType::FLOAT:
            return ExecCompareLeftType<float>(input);
        case DataType::DOUBLE:
            return ExecCompareLeftType<double>(input);
        default:
            PanicInfo(
                DataTypeInvalid,
                fmt::format("unsupported left datatype:{} of compare expr",
                            expr_->left_data_type_));
    }
}

template <typename T>
VectorPtr
PhyCompareFilterExpr::ExecCompareLeftType(OffsetVector* input) {
    switch (expr_->right_data_type_) {
        case DataType::BOOL:
            return ExecCompareRightType<T, bool>(input);
        case DataType::INT8:
            return ExecCompareRightType<T, int8_t>(input);
        case DataType::INT16:
            return ExecCompareRightType<T, int16_t>(input);
        case DataType::INT32:
            return ExecCompareRightType<T, int32_t>(input);
        case DataType::INT64:
            return ExecCompareRightType<T, int64_t>(input);
        case DataType::FLOAT:
            return ExecCompareRightType<T, float>(input);
        case DataType::DOUBLE:
            return ExecCompareRightType<T, double>(input);
        default:
            PanicInfo(
                DataTypeInvalid,
                fmt::format("unsupported right datatype:{} of compare expr",
                            expr_->right_data_type_));
    }
}

template <typename T, typename U>
VectorPtr
PhyCompareFilterExpr::ExecCompareRightType(OffsetVector* input) {
    auto real_batch_size = has_input_ ? input->size() : GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto res_vec = std::make_shared<ColumnVector>(
        TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
    valid_res.set();

    auto expr_type = expr_->op_type_;
    auto execute_sub_batch =
        [ expr_type, this ]<FilterType filter_type = FilterType::pre>(
            const T* left,
            const U* right,
            const int64_t* offsets,
            const int size,
            TargetBitmapView res) {
        switch (expr_type) {
            case proto::plan::GreaterThan: {
                CompareElementFunc<T, U, proto::plan::GreaterThan, filter_type>
                    func;
                func(left, right, size, res, offsets);
                break;
            }
            case proto::plan::GreaterEqual: {
                CompareElementFunc<T, U, proto::plan::GreaterEqual, filter_type>
                    func;
                func(left, right, size, res, offsets);
                break;
            }
            case proto::plan::LessThan: {
                CompareElementFunc<T, U, proto::plan::LessThan, filter_type>
                    func;
                func(left, right, size, res, offsets);
                break;
            }
            case proto::plan::LessEqual: {
                CompareElementFunc<T, U, proto::plan::LessEqual, filter_type>
                    func;
                func(left, right, size, res, offsets);
                break;
            }
            case proto::plan::Equal: {
                CompareElementFunc<T, U, proto::plan::Equal, filter_type> func;
                func(left, right, size, res, offsets);
                break;
            }
            case proto::plan::NotEqual: {
                CompareElementFunc<T, U, proto::plan::NotEqual, filter_type>
                    func;
                func(left, right, size, res, offsets);
                break;
            }
            default:
                PanicInfo(OpTypeInvalid,
                          fmt::format("unsupported operator type for "
                                      "compare column expr: {}",
                                      expr_type));
        }
    };
    int64_t processed_size;
    if (has_input_) {
        processed_size = ProcessBothDataByOffsets<T, U>(
            execute_sub_batch, input, res, valid_res);
    } else {
        processed_size = ProcessBothDataChunks<T, U>(
            execute_sub_batch, input, res, valid_res);
    }
    AssertInfo(processed_size == real_batch_size,
               "internal error: expr processed rows {} not equal "
               "expect batch size {}",
               processed_size,
               real_batch_size);
    return res_vec;
};

}  //namespace exec
}  // namespace milvus
