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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "common/type_c.h"
#include "futures/future_c.h"
#include "segcore/plan_c.h"
#include "segcore/load_index_c.h"
#include "segcore/load_field_data_c.h"

typedef void* CSearchResult;
typedef CProto CRetrieveResult;

//////////////////////////////    two-stage search types    //////////////////////////////
/**
 * @brief Result structure for filter-only execution (Stage 1 of two-stage search)
 * Contains the bitset from scalar filtering and statistics needed for search optimization
 */
typedef struct CFilterResult {
    void* bitset_data;          // Serialized bitset data (caller must free)
    int64_t bitset_size;        // Size of bitset in bytes
    int64_t total_rows;         // Total rows in segment before filtering
    int64_t filtered_count;     // Number of rows passing the filter (bitset cardinality)
} CFilterResult;

//////////////////////////////    common interfaces    //////////////////////////////
CStatus
NewSegment(CCollection collection,
           SegmentType seg_type,
           int64_t segment_id,
           CSegmentInterface* newSegment,
           bool is_sorted_by_pk);

void
DeleteSegment(CSegmentInterface c_segment);

void
ClearSegmentData(CSegmentInterface c_segment);

void
DeleteSearchResult(CSearchResult search_result);

/**
 * @brief Unified async search interface supporting all search modes
 *
 * This function handles three search modes based on parameters:
 * 1. Normal search: c_placeholder_group != nullptr, c_bitset_data == nullptr, filter_only == false
 *    Returns Future<CSearchResult>
 * 2. Filter-only (two-stage stage 1): filter_only == true
 *    Returns Future<CFilterResult*> containing bitset
 * 3. Search with bitset (two-stage stage 2): c_bitset_data != nullptr
 *    Returns Future<CSearchResult> using pre-computed bitset
 *
 * @param c_trace: Tracing context for distributed tracing
 * @param c_segment: Segment to execute search on
 * @param c_plan: Search plan containing filter predicates
 * @param c_placeholder_group: Query vectors (nullptr for filter-only mode)
 * @param timestamp: MVCC timestamp for consistency
 * @param consistency_level: Consistency level for the query
 * @param collection_ttl: Collection TTL timestamp
 * @param c_bitset_data: Pre-computed bitset (nullptr for normal/filter-only mode)
 * @param c_bitset_size: Size of bitset in bytes (0 for normal/filter-only mode)
 * @param filter_only: If true, execute filter-only mode (ignore placeholder_group)
 * @return CFuture* Future that resolves to CSearchResult or CFilterResult* based on mode
 */
CFuture*
AsyncSearch(CTraceContext c_trace,
            CSegmentInterface c_segment,
            CSearchPlan c_plan,
            CPlaceholderGroup c_placeholder_group,
            uint64_t timestamp,
            int32_t consistency_level,
            uint64_t collection_ttl,
            const uint8_t* c_bitset_data,
            int64_t c_bitset_size,
            bool filter_only);

/**
 * @brief Delete a filter result returned by AsyncSearch in filter-only mode
 * @param filter_result: Pointer to CFilterResult to delete
 */
void
DeleteFilterResult(CFilterResult* filter_result);

void
DeleteRetrieveResult(CRetrieveResult* retrieve_result);

CFuture*  // Future<CRetrieveResult>
AsyncRetrieve(CTraceContext c_trace,
              CSegmentInterface c_segment,
              CRetrievePlan c_plan,
              uint64_t timestamp,
              int64_t limit_size,
              bool ignore_non_pk,
              int32_t consistency_level,
              uint64_t collection_ttl);

CFuture*  // Future<CRetrieveResult>
AsyncRetrieveByOffsets(CTraceContext c_trace,
                       CSegmentInterface c_segment,
                       CRetrievePlan c_plan,
                       int64_t* offsets,
                       int64_t len);

int64_t
GetMemoryUsageInBytes(CSegmentInterface c_segment);

int64_t
GetRowCount(CSegmentInterface c_segment);

int64_t
GetDeletedCount(CSegmentInterface c_segment);

int64_t
GetRealCount(CSegmentInterface c_segment);

bool
HasRawData(CSegmentInterface c_segment, int64_t field_id);

bool
HasFieldData(CSegmentInterface c_segment, int64_t field_id);

//////////////////////////////    interfaces for growing segment    //////////////////////////////
CStatus
Insert(CSegmentInterface c_segment,
       int64_t reserved_offset,
       int64_t size,
       const int64_t* row_ids,
       const uint64_t* timestamps,
       const uint8_t* data_info,
       const uint64_t data_info_len);

CStatus
PreInsert(CSegmentInterface c_segment, int64_t size, int64_t* offset);

//////////////////////////////    interfaces for sealed segment    //////////////////////////////
CStatus
LoadFieldData(CSegmentInterface c_segment,
              CLoadFieldDataInfo load_field_data_info);

CStatus
LoadDeletedRecord(CSegmentInterface c_segment,
                  CLoadDeletedRecordInfo deleted_record_info);

CStatus
UpdateSealedSegmentIndex(CSegmentInterface c_segment,
                         CLoadIndexInfo c_load_index_info);

CStatus
LoadTextIndex(CSegmentInterface c_segment,
              const uint8_t* serialized_load_text_index_info,
              const uint64_t len);

CStatus
LoadJsonKeyIndex(CTraceContext c_trace,
                 CSegmentInterface c_segment,
                 const uint8_t* serialied_load_json_key_index_info,
                 const uint64_t len);

CStatus
UpdateFieldRawDataSize(CSegmentInterface c_segment,
                       int64_t field_id,
                       int64_t num_rows,
                       int64_t field_data_size);

// This function is currently used only in test.
// Current implement supports only dropping of non-system fields.
CStatus
DropFieldData(CSegmentInterface c_segment, int64_t field_id);

CStatus
DropSealedSegmentIndex(CSegmentInterface c_segment, int64_t field_id);

CStatus
DropSealedSegmentJSONIndex(CSegmentInterface c_segment,
                           int64_t field_id,
                           const char* nested_path);

CStatus
AddFieldDataInfoForSealed(CSegmentInterface c_segment,
                          CLoadFieldDataInfo c_load_field_data_info);

//////////////////////////////    interfaces for SegmentInterface    //////////////////////////////
CStatus
ExistPk(CSegmentInterface c_segment,
        const uint8_t* raw_ids,
        const uint64_t size,
        bool* results);

CStatus
Delete(CSegmentInterface c_segment,
       int64_t size,
       const uint8_t* ids,
       const uint64_t ids_size,
       const uint64_t* timestamps);

void
RemoveFieldFile(CSegmentInterface c_segment, int64_t field_id);

CStatus
CreateTextIndex(CSegmentInterface c_segment, int64_t field_id);

CStatus
FinishLoad(CSegmentInterface c_segment);

CStatus
ExprResCacheEraseSegment(int64_t segment_id);

#ifdef __cplusplus
}
#endif
