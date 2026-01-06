package tasks

import (
	"bytes"
	"context"
	"fmt"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/querynodev2/cluster"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/searchutil/scheduler"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

var (
	_ scheduler.Task      = &SearchFilterOnlyTask{}
	_ scheduler.MergeTask = &SearchFilterOnlyTask{}
)

// SearchFilterOnlyTask executes filter-only stage of two-stage search.
// It returns filter statistics for search parameter optimization.
// Stage 2 uses normal search (filter is re-executed since it's lightweight).
type SearchFilterOnlyTask struct {
	ctx            context.Context
	collection     *segments.Collection
	segmentManager *segments.Manager
	req            *querypb.SearchRequest
	filterResults  map[int64]*cluster.FilterResult
	notifier       chan error
	serverID       int64

	// Merge-related fields
	merged    bool
	groupSize int64
	nq        int64
	others    []*SearchFilterOnlyTask

	tr           *timerecord.TimeRecorder
	scheduleSpan trace.Span
}

func NewSearchFilterOnlyTask(
	ctx context.Context,
	collection *segments.Collection,
	manager *segments.Manager,
	req *querypb.SearchRequest,
	serverID int64,
) *SearchFilterOnlyTask {
	ctx, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "schedule")
	return &SearchFilterOnlyTask{
		ctx:            ctx,
		collection:     collection,
		segmentManager: manager,
		req:            req,
		filterResults:  make(map[int64]*cluster.FilterResult),
		merged:         false,
		groupSize:      1,
		nq:             req.GetReq().GetNq(),
		notifier:       make(chan error, 1),
		tr:             timerecord.NewTimeRecorderWithTrace(ctx, "searchFilterOnlyTask"),
		scheduleSpan:   span,
		serverID:       serverID,
	}
}

func (t *SearchFilterOnlyTask) Username() string {
	return t.req.Req.GetUsername()
}

func (t *SearchFilterOnlyTask) GetNodeID() int64 {
	return t.serverID
}

func (t *SearchFilterOnlyTask) IsGpuIndex() bool {
	return t.collection.IsGpuIndex()
}

func (t *SearchFilterOnlyTask) PreExecute() error {
	nodeID := strconv.FormatInt(t.GetNodeID(), 10)
	inQueueDuration := t.tr.ElapseSpan()
	inQueueDurationMS := inQueueDuration.Seconds() * 1000

	metrics.QueryNodeSQLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel,
		t.collection.GetDBName(),
		t.collection.GetResourceGroup(),
	).Observe(inQueueDurationMS)

	username := t.Username()
	metrics.QueryNodeSQPerUserLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel,
		username).
		Observe(inQueueDurationMS)

	// Execute merged task's PreExecute.
	for _, subTask := range t.others {
		err := subTask.PreExecute()
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *SearchFilterOnlyTask) Execute() error {
	log := log.Ctx(t.ctx).With(
		zap.Int64("collectionID", t.collection.ID()),
		zap.Int64s("segmentIDs", t.req.GetSegmentIDs()),
	)

	if t.scheduleSpan != nil {
		t.scheduleSpan.End()
	}

	searchReq, err := segcore.NewSearchRequest(t.collection.GetCCollection(), t.req, t.req.GetReq().GetPlaceholderGroup())
	if err != nil {
		return err
	}
	defer searchReq.Delete()

	for _, segID := range t.req.GetSegmentIDs() {
		segment, err := t.segmentManager.Segment.GetAndPinBy(segments.WithID(segID))
		if err != nil {
			log.Warn("Segment not found for filter stage", zap.Int64("segmentID", segID), zap.Error(err))
			continue
		}
		if len(segment) == 0 {
			continue
		}
		seg := segment[0]
		defer t.segmentManager.Segment.Unpin(segment)

		// Use unified Search with filterOnly=true
		searchResult, err := seg.Search(t.ctx, searchReq, true)
		if err != nil {
			log.Warn("Filter-only execution failed", zap.Int64("segmentID", segID), zap.Error(err))
			return err
		}
		defer searchResult.Release()

		// Extract valid count from search result
		t.filterResults[segID] = &cluster.FilterResult{
			SegmentID:  segID,
			ValidCount: searchResult.ValidCount(),
		}
	}

	// Share filter results with merged tasks
	for _, other := range t.others {
		other.filterResults = t.filterResults
	}

	log.Debug("filter-only search completed", zap.Int("results", len(t.filterResults)))
	return nil
}

func (t *SearchFilterOnlyTask) Merge(other *SearchFilterOnlyTask) bool {
	// Check mergeable - filter-only tasks can be merged if they target the same segments
	if t.req.GetReq().GetDbID() != other.req.GetReq().GetDbID() ||
		t.req.GetReq().GetCollectionID() != other.req.GetReq().GetCollectionID() ||
		t.req.GetReq().GetMvccTimestamp() != other.req.GetReq().GetMvccTimestamp() ||
		t.req.GetDmlChannels()[0] != other.req.GetDmlChannels()[0] ||
		t.nq+other.nq > paramtable.Get().QueryNodeCfg.MaxGroupNQ.GetAsInt64() ||
		!funcutil.SliceSetEqual(t.req.GetReq().GetPartitionIDs(), other.req.GetReq().GetPartitionIDs()) ||
		!funcutil.SliceSetEqual(t.req.GetSegmentIDs(), other.req.GetSegmentIDs()) ||
		!bytes.Equal(t.req.GetReq().GetSerializedExprPlan(), other.req.GetReq().GetSerializedExprPlan()) {
		return false
	}

	// Merge
	t.groupSize += other.groupSize
	t.nq += other.nq
	t.others = append(t.others, other)
	other.merged = true

	return true
}

func (t *SearchFilterOnlyTask) MergeWith(other scheduler.Task) bool {
	switch other := other.(type) {
	case *SearchFilterOnlyTask:
		return t.Merge(other)
	}
	return false
}

func (t *SearchFilterOnlyTask) Done(err error) {
	if !t.merged {
		metrics.QueryNodeSearchGroupSize.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.groupSize))
		metrics.QueryNodeSearchGroupNQ.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.nq))
	}
	t.notifier <- err
	for _, other := range t.others {
		other.Done(err)
	}
}

func (t *SearchFilterOnlyTask) Canceled() error {
	return t.ctx.Err()
}

func (t *SearchFilterOnlyTask) Wait() error {
	return <-t.notifier
}

func (t *SearchFilterOnlyTask) NQ() int64 {
	return t.nq
}

// SearchResult returns nil for filter-only task (use FilterResults instead)
func (t *SearchFilterOnlyTask) SearchResult() *internalpb.SearchResults {
	return nil
}

// FilterResults returns the filter results from the task
func (t *SearchFilterOnlyTask) FilterResults() map[int64]*cluster.FilterResult {
	return t.filterResults
}
