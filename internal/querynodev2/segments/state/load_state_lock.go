package state

import (
	"fmt"
	"sync"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
)

type loadStateEnum int

// LoadState represent the state transition of segment.
// LoadStateOnlyMeta: segment is created with meta, but not loaded.
// LoadStateDataLoading: segment is loading data.
// LoadStateDataLoaded: segment is full loaded, ready to be searched or queried.
// LoadStateDataReleasing: segment is releasing data.
// LoadStateReleased: segment is released.
// LoadStateOnlyMeta -> LoadStateDataLoading -> LoadStateDataLoaded -> LoadStateDataReleasing -> (LoadStateReleased or LoadStateOnlyMeta)
const (
	LoadStateOnlyMeta    loadStateEnum = iota
	LoadStateDataLoading               // There will be only one goroutine access segment when loading.
	LoadStateDataLoaded
	LoadStateDataReleasing // There will be only one goroutine access segment when releasing.
	LoadStateReleased
)

// LoadState is the state of segment loading.
func (ls loadStateEnum) String() string {
	switch ls {
	case LoadStateOnlyMeta:
		return "meta"
	case LoadStateDataLoading:
		return "loading-data"
	case LoadStateDataLoaded:
		return "loaded"
	case LoadStateDataReleasing:
		return "releasing-data"
	case LoadStateReleased:
		return "released"
	default:
		return "unknown"
	}
}

// NewLoadStateLock creates a LoadState.
func NewLoadStateLock(state loadStateEnum) *LoadStateLock {
	if state != LoadStateOnlyMeta && state != LoadStateDataLoaded {
		panic(fmt.Sprintf("invalid state for construction of LoadStateLock, %s", state.String()))
	}

	mu := &sync.RWMutex{}
	return &LoadStateLock{
		mu:     mu,
		cv:     sync.Cond{L: mu},
		state:  state,
		refCnt: atomic.NewInt32(0),
	}
}

// LoadStateLock is the state of segment loading.
type LoadStateLock struct {
	mu     *sync.RWMutex
	cv     sync.Cond
	state  loadStateEnum
	refCnt *atomic.Int32
	// ReleaseAll can be called only when refCnt is 0.
	// We need it to be modified when lock is
}

// RLockIfNotReleased locks the segment if the state is not released.
func (ls *LoadStateLock) RLockIf(pred StatePredicate) bool {
	ls.mu.RLock()
	if !pred(ls.state) {
		ls.mu.RUnlock()
		return false
	}
	return true
}

// RUnlock unlocks the segment.
func (ls *LoadStateLock) RUnlock() {
	ls.mu.RUnlock()
}

// PinIfNotReleased pin the segment into memory, avoid ReleaseAll to release it.
func (ls *LoadStateLock) PinIfNotReleased() bool {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	if ls.state == LoadStateReleased {
		return false
	}
	ls.refCnt.Inc()
	return true
}

// Unpin unpin the segment, then segment can be released by ReleaseAll.
func (ls *LoadStateLock) Unpin() {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	newCnt := ls.refCnt.Dec()
	if newCnt < 0 {
		panic("unpin more than pin")
	}
	if newCnt == 0 {
		// notify ReleaseAll to release segment if refcnt is zero.
		ls.cv.Broadcast()
	}
}

// StartLoadData starts load segment data
// Fast fail if segment is not in LoadStateOnlyMeta.
func (ls *LoadStateLock) StartLoadData() (LoadStateLockGuard, error) {
	// only meta can be loaded.
	ls.cv.L.Lock()
	defer ls.cv.L.Unlock()

	if ls.state == LoadStateDataLoaded {
		return nil, nil
	}
	if ls.state != LoadStateOnlyMeta {
		return nil, errors.New("segment is not in LoadStateOnlyMeta, cannot start to loading data")
	}
	ls.state = LoadStateDataLoading
	ls.cv.Broadcast()

	return newLoadStateLockGuard(ls, LoadStateOnlyMeta, LoadStateDataLoaded), nil
}

// StartReleaseData wait until the segment is releasable and starts releasing segment data.
func (ls *LoadStateLock) StartReleaseData() (g LoadStateLockGuard) {
	ls.cv.L.Lock()
	defer ls.cv.L.Unlock()

	ls.waitUntilCanReleaseData()

	switch ls.state {
	case LoadStateDataLoaded:
		ls.state = LoadStateDataReleasing
		ls.cv.Broadcast()
		return newLoadStateLockGuard(ls, LoadStateDataLoaded, LoadStateOnlyMeta)
	case LoadStateOnlyMeta:
		// already transit to target state, do nothing.
		return nil
	case LoadStateReleased:
		// do nothing for empty segment.
		return nil
	default:
		panic(fmt.Sprintf("unreachable code: invalid state when releasing data, %s", ls.state.String()))
	}
}

// StartReleaseAll wait until the segment is releasable and starts releasing all segment.
func (ls *LoadStateLock) StartReleaseAll() (g LoadStateLockGuard) {
	ls.cv.L.Lock()
	defer ls.cv.L.Unlock()

	ls.waitUntilCanReleaseAll()

	switch ls.state {
	case LoadStateDataLoaded:
		ls.state = LoadStateReleased
		ls.cv.Broadcast()
		return newNopLoadStateLockGuard()
	case LoadStateOnlyMeta:
		ls.state = LoadStateReleased
		ls.cv.Broadcast()
		return newNopLoadStateLockGuard()
	case LoadStateReleased:
		// already transit to target state, do nothing.
		return nil
	default:
		panic(fmt.Sprintf("unreachable code: invalid state when releasing data, %s", ls.state.String()))
	}
}

// blockUntilDataLoadedOrReleased blocks until the segment is loaded or released.
func (ls *LoadStateLock) BlockUntilDataLoadedOrReleased() {
	ls.cv.L.Lock()
	defer ls.cv.L.Unlock()

	for ls.state != LoadStateDataLoaded && ls.state != LoadStateReleased {
		ls.cv.Wait()
	}
}

// waitUntilCanReleaseData waits until segment is release data able.
func (ls *LoadStateLock) waitUntilCanReleaseData() {
	state := ls.state
	for state != LoadStateDataLoaded && state != LoadStateOnlyMeta && state != LoadStateReleased {
		ls.cv.Wait()
		state = ls.state
	}
}

// waitUntilCanReleaseAll waits until segment is releasable.
func (ls *LoadStateLock) waitUntilCanReleaseAll() {
	state := ls.state
	for (state != LoadStateDataLoaded && state != LoadStateOnlyMeta && state != LoadStateReleased) || ls.refCnt.Load() != 0 {
		ls.cv.Wait()
		state = ls.state
	}
}

type StatePredicate func(state loadStateEnum) bool

// IsNotReleased checks if the segment is not released.
func IsNotReleased(state loadStateEnum) bool {
	return state != LoadStateReleased
}

// IsDataLoaded checks if the segment is loaded.
func IsDataLoaded(state loadStateEnum) bool {
	return state == LoadStateDataLoaded
}
