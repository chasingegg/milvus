package optimizers

import (
	"context"
	"encoding/json"
	"fmt"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

// QueryHook is the interface for search/query parameter optimizer.
type QueryHook interface {
	Run(map[string]any) error
	Init(string) error
	InitTuningConfig(map[string]string) error
	DeleteTuningConfig(string) error
}

func OptimizeSearchParams(ctx context.Context, req *querypb.SearchRequest, queryHook QueryHook, numSegments int) (*querypb.SearchRequest, error) {
	// no hook applied or disabled, just return
	// if queryHook == nil || !paramtable.Get().AutoIndexConfig.Enable.GetAsBool() {
	// 	req.Req.IsTopkReduce = false
	// 	req.Req.IsRecallEvaluation = false
	// 	return req, nil
	// }

	collectionId := req.GetReq().GetCollectionID()
	log := log.Ctx(ctx).With(zap.Int64("collection", collectionId))

	serializedPlan := req.GetReq().GetSerializedExprPlan()
	// plan not found
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return req, merr.WrapErrParameterInvalid("serialized search plan", "nil")
	}

	channelNum := req.GetTotalChannelNum()
	// not set, change to conservative channel num 1
	if channelNum <= 0 {
		channelNum = 1
	}

	plan := planpb.PlanNode{}
	err := proto.Unmarshal(serializedPlan, &plan)
	if err != nil {
		log.Warn("failed to unmarshal plan", zap.Error(err))
		return nil, merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		// use shardNum * segments num in shard to estimate total segment number
		estSegmentNum := numSegments * int(channelNum)
		metrics.QueryNodeSearchHitSegmentNum.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), fmt.Sprint(collectionId), metrics.SearchLabel).Observe(float64(estSegmentNum))

		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()
		params := map[string]any{
			common.TopKKey:         queryInfo.GetTopk(),
			common.SearchParamKey:  queryInfo.GetSearchParams(),
			common.SegmentNumKey:   estSegmentNum,
			common.WithFilterKey:   withFilter,
			common.DataTypeKey:     int32(plan.GetVectorAnns().GetVectorType()),
			common.WithOptimizeKey: paramtable.Get().AutoIndexConfig.EnableOptimize.GetAsBool() && req.GetReq().GetIsTopkReduce() && queryInfo.GetGroupByFieldId() < 0,
			common.CollectionKey:   req.GetReq().GetCollectionID(),
			common.RecallEvalKey:   req.GetReq().GetIsRecallEvaluation(),
		}
		if withFilter && channelNum > 1 {
			params[common.ChannelNumKey] = channelNum
		}
		searchParamMap := make(map[string]interface{})
		err = json.Unmarshal([]byte(params["search_param"].(string)), &searchParamMap)
		if err != nil {
			return nil, fmt.Errorf("search params in wrong format:%w", err)
		}
		// if searchParamStr = "null", searchParamMap becomes nil
		if searchParamMap == nil {
			return nil, fmt.Errorf("search params in wrong format: it cannot be null")
		}
	
		// for numeric values, json unmarshal will interpret it as float64
		if paramtable.Get().QueryNodeCfg.TwoStageSearchEfRatio.GetAsFloat() > 0 {
			searchParamMap["ef"] = int64(float64(queryInfo.GetTopk()) * paramtable.Get().QueryNodeCfg.TwoStageSearchEfRatio.GetAsFloat())
		} else {
			searchParamMap["ef"] = int64(float64(queryInfo.GetTopk()) * 0.4)
		}
		topk := searchParamMap["ef"].(int64) // ef is the number of candidates to return
		if paramtable.Get().QueryNodeCfg.TwoStageSearchTopkRatio.GetAsFloat() > 0 {
			topk = int64(float64(topk) * paramtable.Get().QueryNodeCfg.TwoStageSearchTopkRatio.GetAsFloat())
		}
		params[common.TopKKey] = topk
		searchParamJson, err := json.Marshal(searchParamMap)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal search params:%w", err)
		}
		params[common.SearchParamKey] = string(searchParamJson)
		// err := queryHook.Run(params)
		// if err != nil {
		// 	log.Warn("failed to execute queryHook", zap.Error(err))
		// 	return nil, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
		// }
		finalTopk := params[common.TopKKey].(int64)
		isTopkReduce := req.GetReq().GetIsTopkReduce() && (finalTopk < queryInfo.GetTopk())
		queryInfo.Topk = finalTopk
		queryInfo.SearchParams = params[common.SearchParamKey].(string)
		serializedExprPlan, err := proto.Marshal(&plan)
		if err != nil {
			log.Warn("failed to marshal optimized plan", zap.Error(err))
			return nil, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
		}
		req.Req.SerializedExprPlan = serializedExprPlan
		req.Req.IsTopkReduce = isTopkReduce
		if isRecallEvaluation, ok := params[common.RecallEvalKey]; ok {
			req.Req.IsRecallEvaluation = isRecallEvaluation.(bool) && queryInfo.GetGroupByFieldId() < 0
		} else {
			req.Req.IsRecallEvaluation = false
		}
		log.Debug("optimized search params done", zap.Any("queryInfo", queryInfo))
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return req, nil
}

// OptimizeSearchParamsWithFilterStats optimizes search parameters using actual filter statistics
// from the two-stage search process. This function is called after stage 1 (filter-only) has
// completed, providing the actual filter selectivity instead of estimates.
//
// Parameters:
//   - ctx: Context for the operation
//   - req: The search request to optimize
//   - queryHook: The query hook for parameter optimization
//   - numSegments: Number of segments being searched
//   - actualFilterRatio: The actual ratio of rows passing the filter (0.0-1.0)
//   - actualFilteredCount: The actual count of rows passing the filter
//   - totalRows: Total rows across all segments before filtering
func OptimizeSearchParamsWithFilterStats(
	ctx context.Context,
	req *querypb.SearchRequest,
	queryHook QueryHook,
	numSegments int,
	actualFilterRatio float64,
	actualFilteredCount int64,
	totalRows int64,
) (*querypb.SearchRequest, error) {
	// no hook applied or disabled, just return
	// if queryHook == nil || !paramtable.Get().AutoIndexConfig.Enable.GetAsBool() {
	// 	req.Req.IsTopkReduce = false
	// 	req.Req.IsRecallEvaluation = false
	// 	return req, nil
	// }

	collectionId := req.GetReq().GetCollectionID()
	log := log.Ctx(ctx).With(
		zap.Int64("collection", collectionId),
		zap.Float64("actualFilterRatio", actualFilterRatio),
		zap.Int64("actualFilteredCount", actualFilteredCount),
		zap.Int64("totalRows", totalRows),
	)

	serializedPlan := req.GetReq().GetSerializedExprPlan()
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return req, merr.WrapErrParameterInvalid("serialized search plan", "nil")
	}

	channelNum := req.GetTotalChannelNum()
	if channelNum <= 0 {
		channelNum = 1
	}

	plan := planpb.PlanNode{}
	err := proto.Unmarshal(serializedPlan, &plan)
	if err != nil {
		log.Warn("failed to unmarshal plan", zap.Error(err))
		return nil, merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		estSegmentNum := numSegments * int(channelNum)
		metrics.QueryNodeSearchHitSegmentNum.WithLabelValues(
			fmt.Sprint(paramtable.GetNodeID()),
			fmt.Sprint(collectionId),
			metrics.SearchLabel,
		).Observe(float64(estSegmentNum))

		// Use actual filter statistics instead of just checking if filter exists
		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()

		params := map[string]any{
			common.TopKKey:         queryInfo.GetTopk(),
			common.SearchParamKey:  queryInfo.GetSearchParams(),
			common.SegmentNumKey:   estSegmentNum,
			common.WithFilterKey:   withFilter,
			common.DataTypeKey:     int32(plan.GetVectorAnns().GetVectorType()),
			common.WithOptimizeKey: paramtable.Get().AutoIndexConfig.EnableOptimize.GetAsBool() && req.GetReq().GetIsTopkReduce() && queryInfo.GetGroupByFieldId() < 0,
			common.CollectionKey:   req.GetReq().GetCollectionID(),
			common.RecallEvalKey:   req.GetReq().GetIsRecallEvaluation(),
			// NEW: Actual filter statistics from two-stage search
			common.FilterRatioKey:       actualFilterRatio,
			common.FilteredRowCountKey:  actualFilteredCount,
			common.TotalRowCountKey:     totalRows,
			common.TwoStageSearchKey:    true, // Flag indicating this is from two-stage search
		}

		if withFilter && channelNum > 1 {
			params[common.ChannelNumKey] = channelNum
		}

		log.Debug("optimizing search params with actual filter stats",
			zap.Int("estSegmentNum", estSegmentNum),
			zap.Bool("withFilter", withFilter),
		)
		searchParamMap := make(map[string]interface{})
		err = json.Unmarshal([]byte(params["search_param"].(string)), &searchParamMap)
		if err != nil {
			return nil, fmt.Errorf("search params in wrong format:%w", err)
		}
		// if searchParamStr = "null", searchParamMap becomes nil
		if searchParamMap == nil {
			return nil, fmt.Errorf("search params in wrong format: it cannot be null")
		}
	
		// for numeric values, json unmarshal will interpret it as float64
		if paramtable.Get().QueryNodeCfg.TwoStageSearchEfRatio.GetAsFloat() > 0 {
			searchParamMap["ef"] = int64(float64(queryInfo.GetTopk()) * paramtable.Get().QueryNodeCfg.TwoStageSearchEfRatio.GetAsFloat())
		} else {
			searchParamMap["ef"] = int64(float64(queryInfo.GetTopk()) / float64(estSegmentNum) * 1.2)
		}
		topk := searchParamMap["ef"].(int64)
		if paramtable.Get().QueryNodeCfg.TwoStageSearchTopkRatio.GetAsFloat() > 0 {
			topk = int64(float64(topk) * paramtable.Get().QueryNodeCfg.TwoStageSearchTopkRatio.GetAsFloat())
		}
		params[common.TopKKey] = topk
		searchParamJson, err := json.Marshal(searchParamMap)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal search params:%w", err)
		}
		params[common.SearchParamKey] = string(searchParamJson)
		// err := queryHook.Run(params)
		// if err != nil {
		// 	log.Warn("failed to execute queryHook with filter stats", zap.Error(err))
		// 	return nil, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
		// }

		finalTopk := params[common.TopKKey].(int64)
		isTopkReduce := req.GetReq().GetIsTopkReduce() && (finalTopk < queryInfo.GetTopk())
		queryInfo.Topk = finalTopk
		queryInfo.SearchParams = params[common.SearchParamKey].(string)

		serializedExprPlan, err := proto.Marshal(&plan)
		if err != nil {
			log.Warn("failed to marshal optimized plan", zap.Error(err))
			return nil, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
		}

		req.Req.SerializedExprPlan = serializedExprPlan
		req.Req.IsTopkReduce = isTopkReduce

		if isRecallEvaluation, ok := params[common.RecallEvalKey]; ok {
			req.Req.IsRecallEvaluation = isRecallEvaluation.(bool) && queryInfo.GetGroupByFieldId() < 0
		} else {
			req.Req.IsRecallEvaluation = false
		}

		log.Debug("optimized search params with filter stats done",
			zap.Any("queryInfo", queryInfo),
			zap.Int64("finalTopk", finalTopk),
		)

	default:
		log.Warn("not supported node type for filter stats optimization",
			zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}

	return req, nil
}
