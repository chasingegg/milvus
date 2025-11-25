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

package delegator

import (
	"context"
	"sync"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/querynodev2/cluster"
	"github.com/milvus-io/milvus/internal/util/searchutil/optimizers"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

// TwoStageSearchConfig holds configuration for two-stage search
type TwoStageSearchConfig struct {
	Enabled bool
	// MinFilterRatio: Only use two-stage search when filter ratio is above this threshold
	// A high filter ratio means the filter is very selective (filters out many rows),
	// which benefits from two-stage optimization
	MinFilterRatio float64
}

// GetTwoStageSearchConfig returns the two-stage search configuration from paramtable
func GetTwoStageSearchConfig() TwoStageSearchConfig {
	return TwoStageSearchConfig{
		Enabled:        paramtable.Get().QueryNodeCfg.TwoStageSearchEnabled.GetAsBool(),
		MinFilterRatio: paramtable.Get().QueryNodeCfg.TwoStageSearchMinFilterRatio.GetAsFloat(),
	}
}

// FilterStats is an alias for cluster.FilterResult for backward compatibility
type FilterStats = cluster.FilterResult

// shouldUseTwoStageSearch determines if two-stage search should be used for this request
func (sd *shardDelegator) shouldUseTwoStageSearch(req *querypb.SearchRequest) bool {
	config := GetTwoStageSearchConfig()
	return config.Enabled

	// Check if the request has a filter predicate
	// We can only benefit from two-stage search if there's a filter
	serializedPlan := req.GetReq().GetSerializedExprPlan()
	if len(serializedPlan) == 0 {
		return false
	}

	// Parse the plan to check if it actually has filter predicates
	plan := &planpb.PlanNode{}
	if err := proto.Unmarshal(serializedPlan, plan); err != nil {
		// If we can't parse the plan, fall back to normal search
		return false
	}

	// Check if the plan has predicates (filter expressions)
	// The plan should be a VectorAnns node for search requests
	if vectorAnns := plan.GetVectorAnns(); vectorAnns != nil {
		if vectorAnns.GetPredicates() == nil {
			// No filter predicates, two-stage search won't help
			return false
		}
	} else if query := plan.GetQuery(); query != nil {
		if query.GetPredicates() == nil {
			return false
		}
	} else {
		// Unknown plan type, skip two-stage search
		return false
	}

	return true
}

// twoStageSearch implements the two-stage search flow:
// Stage 1: Execute filter-only on all segments, collect bitsets and statistics
// Stage 2: Optimize search params using actual filter stats, execute vector search with cached bitsets
func (sd *shardDelegator) twoStageSearch(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	growing []SegmentEntry,
	sealedRowCount map[int64]int64,
) ([]*internalpb.SearchResults, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", sd.collectionID),
		zap.String("channel", sd.vchannelName),
	)
	log.Debug("Starting two-stage search",
		zap.Int("sealedSegments", len(sealed)),
		zap.Int("growingSegments", len(growing)),
	)

	// ==================== STAGE 1: Filter Only ====================
	log.Debug("Starting filter stage")
	filterStats, err := sd.executeFilterStage(ctx, req, sealed, growing, sealedRowCount)
	if err != nil {
		log.Warn("Two-stage search: filter stage failed", zap.Error(err))
		return nil, err
	}

	// Aggregate statistics
	var totalFiltered, totalRows int64
	for _, stat := range filterStats {
		totalFiltered += stat.FilteredCount
		totalRows += stat.TotalRows
	}

	// Calculate actual filter ratio (proportion of rows FILTERED OUT)
	// filterRatio = 0 means no filtering (all rows pass)
	// filterRatio = 1 means all rows filtered out (none pass)
	var actualFilterRatio float64
	if totalRows > 0 {
		actualFilterRatio = float64(totalFiltered) / float64(totalRows)
	}
	totalPassed := totalRows - totalFiltered

	log.Debug("Two-stage search: filter stage completed",
		zap.Int64("totalPassed", totalPassed),
		zap.Int64("totalRows", totalRows),
		zap.Float64("actualFilterRatio", actualFilterRatio),
	)

	// ==================== Optimize with actual stats ====================
	log.Debug("Optimizing search params with actual stats")
	sealedNum := len(filterStats)
	optimizedReq, err := optimizers.OptimizeSearchParamsWithFilterStats(
		ctx, req, sd.queryHook, sealedNum, actualFilterRatio, totalPassed, totalRows)
	if err != nil {
		log.Warn("Two-stage search: failed to optimize search params", zap.Error(err))
		return nil, err
	}

	// ==================== STAGE 2: Vector Search with Cached Bitsets ====================
	log.Debug("Starting vector search stage")
	results, err := sd.executeVectorSearchStage(ctx, optimizedReq, filterStats, sealed, growing, sealedRowCount)
	if err != nil {
		log.Warn("Two-stage search: vector search stage failed", zap.Error(err))
		return nil, err
	}

	log.Debug("Two-stage search completed", zap.Int("results", len(results)))
	return results, nil
}

// executeFilterStage executes stage 1: filter-only on all segments
// Following the same pattern as normal search: organizeSubTask + executeSubTasks
func (sd *shardDelegator) executeFilterStage(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	growing []SegmentEntry,
	sealedRowCount map[int64]int64,
) (map[int64]*FilterStats, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", sd.collectionID),
		zap.String("channel", sd.vchannelName),
	)

	// Use the same pattern as normal search: organizeSubTask + executeSubTasks
	tasks, err := organizeSubTask(ctx, req, sealed, growing, sd, true, sd.modifySearchRequest)
	if err != nil {
		log.Warn("Two-stage search organizeSubTask failed for filter stage", zap.Error(err))
		return nil, err
	}

	// Execute filter-only tasks using worker.SearchFilterOnly
	filterStats := make(map[int64]*FilterStats)
	var mu sync.Mutex
	var firstErr error
	var errOnce sync.Once

	// Execute filter tasks similar to executeSubTasks pattern
	resultCh := make(chan map[int64]*FilterStats, len(tasks))
	errCh := make(chan error, len(tasks))

	for _, task := range tasks {
		go func(t subTask[*querypb.SearchRequest]) {
			worker := t.worker
			if worker == nil {
				var err error
				worker, err = sd.workerManager.GetWorker(ctx, t.targetID)
				if err != nil {
					errCh <- err
					return
				}
			}

			results, err := worker.SearchFilterOnly(ctx, t.req)
			if err != nil {
				st, ok := status.FromError(err)
				if ok && st.Code() == codes.Unavailable {
					sd.markSegmentOffline(t.req.GetSegmentIDs()...)
				}
				errCh <- err
				return
			}
			resultCh <- results
		}(task)
	}

	// Collect results
	for range tasks {
		select {
		case results := <-resultCh:
			mu.Lock()
			for segID, stat := range results {
				filterStats[segID] = stat
			}
			mu.Unlock()
		case err := <-errCh:
			errOnce.Do(func() { firstErr = err })
		}
	}

	if firstErr != nil {
		log.Warn("Two-stage search: filter stage failed", zap.Error(firstErr))
		return nil, firstErr
	}

	log.Debug("Filter stage completed", zap.Int("segments", len(filterStats)))
	return filterStats, nil
}

// executeVectorSearchStage executes stage 2: vector search with cached bitsets
// Following the same pattern as normal search: organizeSubTask + executeSubTasks
func (sd *shardDelegator) executeVectorSearchStage(
	ctx context.Context,
	req *querypb.SearchRequest,
	filterStats map[int64]*FilterStats,
	sealed []SnapshotItem,
	growing []SegmentEntry,
	sealedRowCount map[int64]int64,
) ([]*internalpb.SearchResults, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", sd.collectionID),
		zap.String("channel", sd.vchannelName),
	)

	// Use the same pattern as normal search: organizeSubTask + executeSubTasks
	tasks, err := organizeSubTask(ctx, req, sealed, growing, sd, true, sd.modifySearchRequest)
	if err != nil {
		log.Warn("Two-stage search organizeSubTask failed", zap.Error(err))
		return nil, err
	}

	// Execute sub-tasks just like normal search, but use worker.SearchWithBitset
	results, err := executeSubTasks(ctx, tasks, NewRowCountBasedEvaluator(sealedRowCount),
		func(ctx context.Context, req *querypb.SearchRequest, worker cluster.Worker) (*internalpb.SearchResults, error) {
			resp, err := worker.SearchWithBitset(ctx, req, filterStats)
			st, ok := status.FromError(err)
			if ok && st.Code() == codes.Unavailable {
				sd.markSegmentOffline(req.GetSegmentIDs()...)
			}
			return resp, err
		}, "Search", log)
	if err != nil {
		log.Warn("Two-stage search executeSubTasks failed", zap.Error(err))
		return nil, err
	}

	log.Debug("Vector search stage completed", zap.Int("results", len(results)))
	return results, nil
}
