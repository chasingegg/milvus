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

	"github.com/milvus-io/milvus/internal/util/searchutil/optimizers"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

// FilterStats holds the filter statistics for a segment from stage 1
type FilterStats struct {
	SegmentID     int64
	BitsetData    []byte  // Serialized bitset (1 = filtered out, 0 = pass)
	TotalRows     int64   // Total rows in segment before filtering
	FilteredCount int64   // Number of rows passing the filter
	FilterRatio   float64 // FilteredCount / TotalRows
}

// TwoStageSearchConfig holds configuration for two-stage search
type TwoStageSearchConfig struct {
	Enabled bool
	// MinFilterRatio: Only use two-stage search when filter ratio is below this threshold
	// A low filter ratio means the filter is very selective, which benefits from optimization
	MinFilterRatio float64
}

// GetTwoStageSearchConfig returns the two-stage search configuration from paramtable
func GetTwoStageSearchConfig() TwoStageSearchConfig {
	return TwoStageSearchConfig{
		Enabled:        paramtable.Get().QueryNodeCfg.TwoStageSearchEnabled.GetAsBool(),
		MinFilterRatio: paramtable.Get().QueryNodeCfg.TwoStageSearchMinFilterRatio.GetAsFloat(),
	}
}

// shouldUseTwoStageSearch determines if two-stage search should be used for this request
func (sd *shardDelegator) shouldUseTwoStageSearch(req *querypb.SearchRequest) bool {
	config := GetTwoStageSearchConfig()
	if !config.Enabled {
		return false
	}

	// Check if the request has a filter predicate
	// We can only benefit from two-stage search if there's a filter
	// The filter existence is determined by the presence of predicates in the plan
	// For now, we'll check a simple heuristic - if SerializedExprPlan is non-empty
	// and larger than a minimum size (indicating non-trivial filter)
	if len(req.GetReq().GetSerializedExprPlan()) == 0 {
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
	filterStats, err := sd.executeFilterStage(ctx, req, sealed, growing)
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

	// Calculate actual filter ratio
	var actualFilterRatio float64
	if totalRows > 0 {
		actualFilterRatio = float64(totalFiltered) / float64(totalRows)
	}

	log.Debug("Two-stage search: filter stage completed",
		zap.Int64("totalFiltered", totalFiltered),
		zap.Int64("totalRows", totalRows),
		zap.Float64("actualFilterRatio", actualFilterRatio),
	)

	// Check if filter ratio is above threshold - if so, fall back to normal search
	config := GetTwoStageSearchConfig()
	if actualFilterRatio > config.MinFilterRatio {
		log.Debug("Two-stage search: filter ratio above threshold, falling back to normal search",
			zap.Float64("actualFilterRatio", actualFilterRatio),
			zap.Float64("threshold", config.MinFilterRatio),
		)
		// Fall back to normal search path
		return sd.search(ctx, req, sealed, growing, sealedRowCount)
	}

	// ==================== Optimize with actual stats ====================
	sealedNum := len(filterStats)
	optimizedReq, err := optimizers.OptimizeSearchParamsWithFilterStats(
		ctx, req, sd.queryHook, sealedNum, actualFilterRatio, totalFiltered, totalRows)
	if err != nil {
		log.Warn("Two-stage search: failed to optimize search params", zap.Error(err))
		return nil, err
	}

	// ==================== STAGE 2: Vector Search with Cached Bitsets ====================
	results, err := sd.executeVectorSearchStage(ctx, optimizedReq, filterStats, sealed, growing, sealedRowCount)
	if err != nil {
		log.Warn("Two-stage search: vector search stage failed", zap.Error(err))
		return nil, err
	}

	log.Debug("Two-stage search completed", zap.Int("results", len(results)))
	return results, nil
}

// executeFilterStage executes stage 1: filter-only on all segments
func (sd *shardDelegator) executeFilterStage(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	growing []SegmentEntry,
) (map[int64]*FilterStats, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", sd.collectionID),
		zap.String("channel", sd.vchannelName),
	)

	filterStats := make(map[int64]*FilterStats)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once

	// Create search plan from request (shared for all segments)
	// Use the underlying CCollection from the segments.Collection wrapper
	ccollection := sd.collection.GetCCollection()
	searchReq, err := segcore.NewSearchRequest(ccollection, req, req.GetReq().GetPlaceholderGroup())
	if err != nil {
		return nil, err
	}
	plan := searchReq.Plan()

	// Process sealed segments
	for _, item := range sealed {
		for _, seg := range item.Segments {
			wg.Add(1)
			go func(segID int64, nodeID int64) {
				defer wg.Done()

				// Get the segment and execute filter-only
				segment := sd.segmentManager.Get(segID)
				if segment == nil {
					log.Warn("Segment not found for filter stage", zap.Int64("segmentID", segID))
					return
				}

				// Execute filter-only
				filterResult, err := segment.SearchFilterOnly(
					ctx,
					plan,
					req.GetReq().GetMvccTimestamp(),
					int32(req.GetReq().GetConsistencyLevel()),
				)
				if err != nil {
					log.Warn("Filter-only execution failed",
						zap.Int64("segmentID", segID),
						zap.Error(err))
					errOnce.Do(func() { firstErr = err })
					return
				}

				// Store result
				var filterRatio float64
				if filterResult.TotalRows > 0 {
					filterRatio = float64(filterResult.FilteredCount) / float64(filterResult.TotalRows)
				}

				mu.Lock()
				filterStats[segID] = &FilterStats{
					SegmentID:     segID,
					BitsetData:    filterResult.BitsetData,
					TotalRows:     filterResult.TotalRows,
					FilteredCount: filterResult.FilteredCount,
					FilterRatio:   filterRatio,
				}
				mu.Unlock()
			}(seg.SegmentID, seg.NodeID)
		}
	}

	// Process growing segments similarly
	for _, seg := range growing {
		wg.Add(1)
		go func(segID int64) {
			defer wg.Done()

			segment := sd.segmentManager.Get(segID)
			if segment == nil {
				log.Warn("Growing segment not found for filter stage", zap.Int64("segmentID", segID))
				return
			}

			filterResult, err := segment.SearchFilterOnly(
				ctx,
				plan,
				req.GetReq().GetMvccTimestamp(),
				int32(req.GetReq().GetConsistencyLevel()),
			)
			if err != nil {
				log.Warn("Filter-only execution failed for growing segment",
					zap.Int64("segmentID", segID),
					zap.Error(err))
				errOnce.Do(func() { firstErr = err })
				return
			}

			var filterRatio float64
			if filterResult.TotalRows > 0 {
				filterRatio = float64(filterResult.FilteredCount) / float64(filterResult.TotalRows)
			}

			mu.Lock()
			filterStats[segID] = &FilterStats{
				SegmentID:     segID,
				BitsetData:    filterResult.BitsetData,
				TotalRows:     filterResult.TotalRows,
				FilteredCount: filterResult.FilteredCount,
				FilterRatio:   filterRatio,
			}
			mu.Unlock()
		}(seg.SegmentID)
	}

	wg.Wait()

	// Clean up the shared search request
	searchReq.Delete()

	if firstErr != nil {
		return nil, firstErr
	}

	return filterStats, nil
}

// executeVectorSearchStage executes stage 2: vector search with cached bitsets
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

	var results []*internalpb.SearchResults
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once

	// Get the underlying CCollection
	ccollection := sd.collection.GetCCollection()

	// Process sealed segments
	for _, item := range sealed {
		for _, seg := range item.Segments {
			wg.Add(1)
			go func(segID int64) {
				defer wg.Done()

				stats, ok := filterStats[segID]
				if !ok {
					log.Warn("No filter stats for segment", zap.Int64("segmentID", segID))
					return
				}

				segment := sd.segmentManager.Get(segID)
				if segment == nil {
					log.Warn("Segment not found for vector search stage", zap.Int64("segmentID", segID))
					return
				}

				// Create search request for this segment
				searchReq, err := segcore.NewSearchRequest(ccollection, req, req.GetReq().GetPlaceholderGroup())
				if err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
				defer searchReq.Delete()

				// Execute vector search with pre-computed bitset
				searchResult, err := segment.SearchWithBitset(ctx, searchReq, stats.BitsetData)
				if err != nil {
					log.Warn("Vector search with bitset failed",
						zap.Int64("segmentID", segID),
						zap.Error(err))
					errOnce.Do(func() { firstErr = err })
					return
				}
				defer searchResult.Release()

				// For now, we skip result conversion - this needs to be properly integrated
				// with the reduce/merge logic. The SearchResult needs to be processed through
				// ReduceSearchResultsAndFillData to get the final blob.
				// This is a placeholder - actual implementation needs to integrate with
				// the existing reduce pipeline.
				_ = searchResult

				mu.Lock()
				// Placeholder result - actual implementation needs proper result handling
				results = append(results, &internalpb.SearchResults{
					MetricType: req.GetReq().GetMetricType(),
				})
				mu.Unlock()
			}(seg.SegmentID)
		}
	}

	// Process growing segments similarly
	for _, seg := range growing {
		wg.Add(1)
		go func(segID int64) {
			defer wg.Done()

			stats, ok := filterStats[segID]
			if !ok {
				log.Warn("No filter stats for growing segment", zap.Int64("segmentID", segID))
				return
			}

			segment := sd.segmentManager.Get(segID)
			if segment == nil {
				log.Warn("Growing segment not found for vector search stage", zap.Int64("segmentID", segID))
				return
			}

			searchReq, err := segcore.NewSearchRequest(ccollection, req, req.GetReq().GetPlaceholderGroup())
			if err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}
			defer searchReq.Delete()

			searchResult, err := segment.SearchWithBitset(ctx, searchReq, stats.BitsetData)
			if err != nil {
				log.Warn("Vector search with bitset failed for growing segment",
					zap.Int64("segmentID", segID),
					zap.Error(err))
				errOnce.Do(func() { firstErr = err })
				return
			}
			defer searchResult.Release()

			// Placeholder - actual implementation needs proper result handling
			_ = searchResult

			mu.Lock()
			results = append(results, &internalpb.SearchResults{
				MetricType: req.GetReq().GetMetricType(),
			})
			mu.Unlock()
		}(seg.SegmentID)
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return results, nil
}
