package delegator

import (
	"fmt"
	"sort"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/distance"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

type SegmentDistanceStruct struct {
	segment  SegmentEntry
	distance float32
}

// set efs for each segment
func (sd *shardDelegator) OptimizeSearchParamHelper2(req *querypb.SearchRequest, distanceArrNq [][]SegmentDistanceStruct, segmentIDMap map[UniqueID]int, sealedNum int) error {
	serializedPlan := req.GetReq().GetSerializedExprPlan()
	filterRatio := req.GetReq().GetClusterBasedFilterRate()
	algo := req.GetReq().GetUseClusterInfo()

	// plan not found
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return merr.WrapErrParameterInvalid("serialized search plan", "nil")
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
		return merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		// use shardNum * segments num in shard to estimate total segment number
		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()

		nq := len(distanceArrNq)
		if nq != 1 {
			log.Warn("current we only support nq=1")
			return fmt.Errorf("current we only support nq=1")
		}
	
		num := len(distanceArrNq[0])
		// fmt.Println("num " + fmt.Sprint(num))

		distanceArr := make([]float32, 0, num)
		clusterSize := make([]int64, 0, num)
		for _, d := range distanceArrNq[0] {
			distanceArr = append(distanceArr, d.distance)
			clusterSize = append(clusterSize, d.segment.ClusterSize)
		}
		estSegmentNum := sealedNum * int(channelNum)
		// generate different search params for a single query
		if algo == 4 || algo == 5 || algo == 6 {
			params := map[string]any{
				common.TopKKey:        req.GetReq().GetTopk(),
				common.SearchParamKey: queryInfo.GetSearchParams(),
				common.SegmentNumKey:  estSegmentNum,
				common.WithFilterKey:  withFilter,
				common.CollectionKey:  req.GetReq().GetCollectionID(),
				common.ClusterRatio:   filterRatio,
				common.ClusterDis:     distanceArr,
				common.ClusterSize:    clusterSize,
				common.ClusterAlpha:   req.GetReq().GetClusterAlpha(),
				common.ClusterParam:   req.GetReq().GetClusterSearchParamList(),
				common.MetricTypeKey:  req.GetReq().GetMetricType(),
				common.Algo:           algo,
			}
			// searchParam, Topk, list size is equal to what we want
			searchParamArr, TopkArray, err := sd.queryHook.Optimize2(params)
			if err != nil {
				log.Warn("failed to execute queryHook", zap.Error(err))
				return merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
			}
			efs := make([]int64, num, num)  // not set, ef = 0
			for i, searchParam := range searchParamArr {
				sid := distanceArrNq[0][i].segment.SegmentID
				efs[segmentIDMap[sid]] = searchParam
			}
			req.Efs = efs
			log.Debug("optimize output", zap.Any("searchParamArr", searchParamArr), zap.Any("topArray", TopkArray), zap.Any("cluster size", clusterSize), zap.Any("efs", efs))
		}
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return nil
}

// generate a plan array for each segment
// input: distance array, segment num
// output: plan array, final segment num
func (sd *shardDelegator) OptimizeSearchParamHelper(req *querypb.SearchRequest, distanceArrNq [][]SegmentDistanceStruct, sealedNum int) ([][]byte, int, error){
	serializedPlan := req.GetReq().GetSerializedExprPlan()
	filterRatio := req.GetReq().GetClusterBasedFilterRate()
	algo := req.GetReq().GetUseClusterInfo()

	// plan not found
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return nil, 0, merr.WrapErrParameterInvalid("serialized search plan", "nil")
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
		return nil, 0, merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		// use shardNum * segments num in shard to estimate total segment number
		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()

		// original algo for optimize search param
		// algo=0 take all segments into account
		// algo=1 take all (segments * filter ratio) into account
		// algo=2 equals algo=1
		// these three methods generate the same search param for a single query
		if algo == 0 || algo == 1 || algo == 2 || algo == 3 {
			estSegmentNum := sealedNum * int(channelNum)

			distanceArr := make([]float32, 0, estSegmentNum)
			clusterSize := make([]int64, 0, estSegmentNum)
			if distanceArrNq != nil {
				for _, d := range distanceArrNq[0] {
					distanceArr = append(distanceArr, d.distance)
					clusterSize = append(clusterSize, d.segment.ClusterSize)
				}
			}
			if algo != 0 {
				estSegmentNum = int(filterRatio * float32(sealedNum)) * int(channelNum)
			}

			params := map[string]any{
				common.TopKKey:        req.GetReq().GetTopk(),
				common.SearchParamKey: queryInfo.GetSearchParams(),
				common.SegmentNumKey:  estSegmentNum,
				// common.WithFilterKey:  withFilter,
				common.ClusterRatio:   filterRatio,
				common.ClusterDis:     distanceArr,
				common.ClusterSize:    clusterSize,
				common.MetricTypeKey:  req.GetReq().GetMetricType(),
				common.CollectionKey:  req.GetReq().GetCollectionID(),
				common.Algo:           algo,
			}
			err := sd.queryHook.Run(params)
			if err != nil {
				log.Warn("failed to execute queryHook", zap.Error(err))
				return nil, 0, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
			}
			// queryInfo.Topk = params[common.TopKKey].(int64)
			queryInfo.SearchParams = params[common.SearchParamKey].(string)
			sp, err := proto.Marshal(&plan)
			if err != nil {
				log.Warn("failed to marshal optimized plan", zap.Error(err))
				return nil, 0, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
			}
			log.Debug("optimize output", zap.Any("searchParamArr", queryInfo.SearchParams), zap.Any("cluster size", clusterSize))

			return [][]byte{sp}, params["selectNum"].(int), nil
		} else {   // will be deprecated if performance is not good
			nq := len(distanceArrNq)
			if nq != 1 {
				log.Warn("current we only support nq=1")
				return nil, 0, fmt.Errorf("current we only support nq=1")
			}
		
			num := len(distanceArrNq[0])
			distanceArr := make([]float32, 0, num)
			for _, d := range distanceArrNq[0] {
				distanceArr = append(distanceArr, d.distance)
			}
			// generate different search params for different segments for a single query
			// 
			if algo == 10 {
				serializedPlanArr := make([][]byte, 0, num)

				params := map[string]any{
					common.TopKKey:        req.GetReq().GetTopk(),
					common.SearchParamKey: queryInfo.GetSearchParams(),
					common.SegmentNumKey:  num * int(channelNum),
					common.WithFilterKey:  withFilter,
					common.CollectionKey:  req.GetReq().GetCollectionID(),
					common.ClusterRatio:   filterRatio,
					common.ClusterDis:     distanceArr,
					common.Algo:           algo,
				}
				// searchParam, Topk, list size is equal to what we want
				searchParamArr, TopkArray, err := sd.queryHook.Optimize(params)
				log.Debug("optimize output", zap.Any("searchParamArr", searchParamArr), zap.Any("topArray", TopkArray))
				if err != nil {
					log.Warn("failed to execute queryHook", zap.Error(err))
					return nil, 0, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
				}
				for _, searchParam := range searchParamArr {
					// queryInfo.Topk = TopkArray[i]
					queryInfo.SearchParams = searchParam
					sp, err := proto.Marshal(&plan)
					if err != nil {
						log.Warn("failed to marshal optimized plan", zap.Error(err))
						return nil, 0, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
					}
					serializedPlanArr = append(serializedPlanArr, sp)
					// req.Req.SerializedExprPlan = serializedExprPlan
					log.Debug("optimized search params done", zap.Any("queryInfo", queryInfo))
				}
				return serializedPlanArr, len(serializedPlanArr), nil
			}
		} 
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return nil, 0, fmt.Errorf("wrong path")
}

// return req, snapshotItem, serializedPlans, error
func (sd *shardDelegator) OptimizeSearchParam(req *querypb.SearchRequest, sealeds []SnapshotItem, sealedNum int) (*querypb.SearchRequest, []SnapshotItem, [][]byte, error) {
	algo := req.GetReq().GetUseClusterInfo()
	// no clustering based optimization
	if algo <= 0 {
		log.Debug("skip optimize based on clustering info")
		serializedPlan, _, err := sd.OptimizeSearchParamHelper(req, nil, sealedNum)
		if err != nil {
			return req, sealeds, nil, err
		}
		req.Req.SerializedExprPlan = serializedPlan[0]
		return req, sealeds, nil, nil
	}

	dim := req.GetReq().GetDim()
	metricType := req.GetReq().GetMetricType()
	var phg commonpb.PlaceholderGroup
	err := proto.Unmarshal(req.GetReq().GetPlaceholderGroup(), &phg)
	if err != nil {
		fmt.Println("Error:", err)
	}
	log.Debug("optimizeSearchBasedOnClustering",
		zap.String("metricType", metricType),
		zap.Int32("dim", dim),
		zap.Int("length", len(phg.GetPlaceholders())),
		zap.Any("phg", phg))

	vectorsBytes := phg.GetPlaceholders()[0].GetValues()
	searchVectors := make([][]float32, len(vectorsBytes))
	for i, vectorBytes := range vectorsBytes {
		searchVectors[i] = Deserialize(vectorBytes)
	}

	var segments = make([]SegmentEntry, 0)
	for _, sealed := range sealeds {
		segments = append(segments, sealed.Segments...)
	}
	segmentIDMap := make(map[UniqueID]int, 0)
	for i, segment := range segments {
		segmentIDMap[segment.SegmentID] = i
	}

	// vector segment distances [[(segmentEntry1, 23), (segmentEntry2, 12), (segmentEntry3, 48)]]
	var vectorSegmentDistances = make([][]SegmentDistanceStruct, 0)
	for _, searchVector := range searchVectors {
		var vectorSegmentDistance = make([]SegmentDistanceStruct, 0)
		for _, segment := range segments {
			if segment.Params != nil {
				distance, err := distance.CalcFloatDistance(int64(dim), searchVector, segment.Params, metricType)
				if err != nil {
					fmt.Println("Error:", err)
				}
				log.Debug("distance between searchVector and cluster center", zap.Int64("segmentID", segment.SegmentID), zap.Float32s("distance", distance))
				vectorSegmentDistance = append(vectorSegmentDistance, SegmentDistanceStruct{segment: segment, distance: distance[0]})
			} else {
				vectorSegmentDistance = append(vectorSegmentDistance, SegmentDistanceStruct{segment: segment, distance: float32(0.0)})
			}
		}
		vectorSegmentDistances = append(vectorSegmentDistances, vectorSegmentDistance)
	}
	log.Debug("vector segment distances", zap.Any("vectorSegmentDistances", vectorSegmentDistances))

	// filtered := make(map[UniqueID]SegmentEntry, 0)
	for _, vectorSegmentDistance := range vectorSegmentDistances {
		sort.SliceStable(vectorSegmentDistance, func(i, j int) bool {
			if metricType == "L2" {
				return vectorSegmentDistance[i].distance < vectorSegmentDistance[j].distance
			} else {
				return vectorSegmentDistance[i].distance > vectorSegmentDistance[j].distance
			}
		}) // sort by distance
	}
	// for _, vectorSegmentDistance := range vectorSegmentDistances[0] {
	// 	log.Debug("vector segment distances", zap.Any("vectors", vectorSegmentDistance.distance))
	// }
	// log.Debug("filtered segments", zap.Int("before", len(segments)), zap.Int("after", len(filtered)), zap.Any("segments", filtered))

	// vectorsegmentDistances 排好序的{segment, distance} pair
	if algo == 4 || algo == 5 || algo == 6 {
		err := sd.OptimizeSearchParamHelper2(req, vectorSegmentDistances, segmentIDMap, sealedNum)
		if err != nil {
			return nil, nil, nil, err
		}
		return req, sealeds, nil, nil
	}

	sp, segmentNum, err := sd.OptimizeSearchParamHelper(req, vectorSegmentDistances, sealedNum)

	log.Debug("after optimize", zap.Int("segment num", segmentNum), zap.Int("whole segment num", sealedNum))
	if err != nil {
		return nil, nil, nil, err
	}

	// merge to SnapshotItem
	nodeSegmentsArr := make([]SnapshotItem, 0)
	
	// all segment in one queryNode use the same search param
	// merge segments in the same node into one SnapshotItem
	if algo == 1 || algo == 2 || algo == 3 {
		nodeSegments := make(map[int64]SnapshotItem, 0)
		for _, segment := range vectorSegmentDistances[0][:segmentNum] {
			if _, ok := nodeSegments[segment.segment.NodeID]; ok {
				snapshot := nodeSegments[segment.segment.NodeID]
				segments := append(snapshot.Segments, segment.segment)
				nodeSegments[segment.segment.NodeID] = SnapshotItem{
					NodeID:   segment.segment.NodeID,
					Segments: segments,
				}
			} else {
				nodeSegments[segment.segment.NodeID] = SnapshotItem{
					NodeID:   segment.segment.NodeID,
					Segments: []SegmentEntry{segment.segment},
				}
			}
		}
		for _, snapshot := range nodeSegments {
			nodeSegmentsArr = append(nodeSegmentsArr, snapshot)
		}
	} else if algo == 10 {  // each segment as a SnapshotItem 
		for _, segment := range vectorSegmentDistances[0][:segmentNum] {
			nodeSegmentsArr = append(nodeSegmentsArr, SnapshotItem{
				NodeID: segment.segment.NodeID,
				Segments: []SegmentEntry{segment.segment},
			})
		}
	} else {
		log.Error("not supported algo!", zap.Int("aglo", int(algo)))
	}

	log.Debug("optimizeSearchBasedOnClustering",
		zap.Any("req", req), zap.Any("sealed", sealeds))

	return req, nodeSegmentsArr, sp, nil
}