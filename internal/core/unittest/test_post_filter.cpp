// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>
#include "common/Schema.h"
#include "query/Plan.h"
#include "segcore/SegmentSealedImpl.h"
#include "segcore/reduce_c.h"
#include "segcore/plan_c.h"
#include "segcore/segment_c.h"
#include "test_utils/DataGen.h"
#include "test_utils/c_api_test_utils.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;
using namespace milvus::storage;
using namespace milvus::tracer;

const char* METRICS_TYPE = "metric_type";

TEST(GroupBY, SealedIndex) {
    using namespace milvus;
    using namespace milvus::query;
    using namespace milvus::segcore;

    //0. prepare schema
    int dim = 64;
    auto schema = std::make_shared<Schema>();
    auto vec_fid = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto str_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto bool_fid = schema->AddDebugField("bool", DataType::BOOL);
    schema->set_primary_field_id(str_fid);
    auto segment = CreateSealedSegment(schema);
    size_t N = 50;

    //2. load raw data
    auto raw_data = DataGen(schema, N, 42, 0, 8, 10, false, false);
    auto fields = schema->get_fields();
    for (auto field_data : raw_data.raw_->fields_data()) {
        int64_t field_id = field_data.field_id();

        auto info = FieldDataInfo(field_data.field_id(), N);
        auto field_meta = fields.at(FieldId(field_id));
        info.channel->push(
            CreateFieldDataFromDataArray(N, &field_data, field_meta));
        info.channel->close();

        segment->LoadFieldData(FieldId(field_id), info);
    }
    prepareSegmentSystemFieldData(segment, N, raw_data);

    //3. load index
    auto vector_data = raw_data.get_col<float>(vec_fid);
    auto indexing = GenVecIndexing(
        N, dim, vector_data.data(), knowhere::IndexEnum::INDEX_HNSW);
    LoadIndexInfo load_index_info;
    load_index_info.field_id = vec_fid.get();
    load_index_info.index = std::move(indexing);
    load_index_info.index_params[METRICS_TYPE] = knowhere::metric::L2;
    segment->LoadIndex(load_index_info);
    int topK = 15;
    int group_size = 3;

    //4. search group by int8
    {
        const char* raw_plan = R"(vector_anns: <
                                        field_id: 100
                                        query_info: <
                                          topk: 15
                                          metric_type: "L2"
                                          search_params: "{\"ef\": 10}"
                                          group_by_field_id: 101
                                          group_size: 3
                                        >
                                        placeholder_tag: "$0"

         >)";
        proto::plan::PlanNode plan_node;
        auto ok =
            google::protobuf::TextFormat::ParseFromString(raw_plan, &plan_node);
        auto plan = CreateSearchPlanFromPlanNode(*schema, plan_node);
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), 1L << 63);
        CheckGroupBySearchResult(*search_result, topK, num_queries, false);

        auto& group_by_values = search_result->group_by_values_.value();
        ASSERT_EQ(20, group_by_values.size());
        //as the total data is 0,0,....6,6, so there will be 7 buckets with [3,3,3,3,3,3,2] items respectively
        //so there will be 20 items returned

        int size = group_by_values.size();
        std::unordered_map<int8_t, int> i8_map;
        float lastDistance = 0.0;
        for (size_t i = 0; i < size; i++) {
            if (std::holds_alternative<int8_t>(group_by_values[i])) {
                int8_t g_val = std::get<int8_t>(group_by_values[i]);
                i8_map[g_val] += 1;
                ASSERT_TRUE(i8_map[g_val] <= group_size);
                //for every group, the number of hits should not exceed group_size
                auto distance = search_result->distances_.at(i);
                ASSERT_TRUE(
                    lastDistance <=
                    distance);  //distance should be decreased as metrics_type is L2
                lastDistance = distance;
            }
        }
        ASSERT_TRUE(i8_map.size() <= topK);
        ASSERT_TRUE(i8_map.size() == 7);
    }
}
