// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_flusher

import (
	wal "github.com/milvus-io/milvus/internal/streamingnode/server/wal"
	mock "github.com/stretchr/testify/mock"
)

// MockFlusher is an autogenerated mock type for the Flusher type
type MockFlusher struct {
	mock.Mock
}

type MockFlusher_Expecter struct {
	mock *mock.Mock
}

func (_m *MockFlusher) EXPECT() *MockFlusher_Expecter {
	return &MockFlusher_Expecter{mock: &_m.Mock}
}

// RegisterPChannel provides a mock function with given fields: pchannel, w
func (_m *MockFlusher) RegisterPChannel(pchannel string, w wal.WAL) error {
	ret := _m.Called(pchannel, w)

	if len(ret) == 0 {
		panic("no return value specified for RegisterPChannel")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(string, wal.WAL) error); ok {
		r0 = rf(pchannel, w)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockFlusher_RegisterPChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RegisterPChannel'
type MockFlusher_RegisterPChannel_Call struct {
	*mock.Call
}

// RegisterPChannel is a helper method to define mock.On call
//   - pchannel string
//   - w wal.WAL
func (_e *MockFlusher_Expecter) RegisterPChannel(pchannel interface{}, w interface{}) *MockFlusher_RegisterPChannel_Call {
	return &MockFlusher_RegisterPChannel_Call{Call: _e.mock.On("RegisterPChannel", pchannel, w)}
}

func (_c *MockFlusher_RegisterPChannel_Call) Run(run func(pchannel string, w wal.WAL)) *MockFlusher_RegisterPChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(wal.WAL))
	})
	return _c
}

func (_c *MockFlusher_RegisterPChannel_Call) Return(_a0 error) *MockFlusher_RegisterPChannel_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockFlusher_RegisterPChannel_Call) RunAndReturn(run func(string, wal.WAL) error) *MockFlusher_RegisterPChannel_Call {
	_c.Call.Return(run)
	return _c
}

// RegisterVChannel provides a mock function with given fields: vchannel, _a1
func (_m *MockFlusher) RegisterVChannel(vchannel string, _a1 wal.WAL) {
	_m.Called(vchannel, _a1)
}

// MockFlusher_RegisterVChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RegisterVChannel'
type MockFlusher_RegisterVChannel_Call struct {
	*mock.Call
}

// RegisterVChannel is a helper method to define mock.On call
//   - vchannel string
//   - _a1 wal.WAL
func (_e *MockFlusher_Expecter) RegisterVChannel(vchannel interface{}, _a1 interface{}) *MockFlusher_RegisterVChannel_Call {
	return &MockFlusher_RegisterVChannel_Call{Call: _e.mock.On("RegisterVChannel", vchannel, _a1)}
}

func (_c *MockFlusher_RegisterVChannel_Call) Run(run func(vchannel string, _a1 wal.WAL)) *MockFlusher_RegisterVChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string), args[1].(wal.WAL))
	})
	return _c
}

func (_c *MockFlusher_RegisterVChannel_Call) Return() *MockFlusher_RegisterVChannel_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFlusher_RegisterVChannel_Call) RunAndReturn(run func(string, wal.WAL)) *MockFlusher_RegisterVChannel_Call {
	_c.Call.Return(run)
	return _c
}

// Start provides a mock function with given fields:
func (_m *MockFlusher) Start() {
	_m.Called()
}

// MockFlusher_Start_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Start'
type MockFlusher_Start_Call struct {
	*mock.Call
}

// Start is a helper method to define mock.On call
func (_e *MockFlusher_Expecter) Start() *MockFlusher_Start_Call {
	return &MockFlusher_Start_Call{Call: _e.mock.On("Start")}
}

func (_c *MockFlusher_Start_Call) Run(run func()) *MockFlusher_Start_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockFlusher_Start_Call) Return() *MockFlusher_Start_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFlusher_Start_Call) RunAndReturn(run func()) *MockFlusher_Start_Call {
	_c.Call.Return(run)
	return _c
}

// Stop provides a mock function with given fields:
func (_m *MockFlusher) Stop() {
	_m.Called()
}

// MockFlusher_Stop_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Stop'
type MockFlusher_Stop_Call struct {
	*mock.Call
}

// Stop is a helper method to define mock.On call
func (_e *MockFlusher_Expecter) Stop() *MockFlusher_Stop_Call {
	return &MockFlusher_Stop_Call{Call: _e.mock.On("Stop")}
}

func (_c *MockFlusher_Stop_Call) Run(run func()) *MockFlusher_Stop_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockFlusher_Stop_Call) Return() *MockFlusher_Stop_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFlusher_Stop_Call) RunAndReturn(run func()) *MockFlusher_Stop_Call {
	_c.Call.Return(run)
	return _c
}

// UnregisterPChannel provides a mock function with given fields: pchannel
func (_m *MockFlusher) UnregisterPChannel(pchannel string) {
	_m.Called(pchannel)
}

// MockFlusher_UnregisterPChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UnregisterPChannel'
type MockFlusher_UnregisterPChannel_Call struct {
	*mock.Call
}

// UnregisterPChannel is a helper method to define mock.On call
//   - pchannel string
func (_e *MockFlusher_Expecter) UnregisterPChannel(pchannel interface{}) *MockFlusher_UnregisterPChannel_Call {
	return &MockFlusher_UnregisterPChannel_Call{Call: _e.mock.On("UnregisterPChannel", pchannel)}
}

func (_c *MockFlusher_UnregisterPChannel_Call) Run(run func(pchannel string)) *MockFlusher_UnregisterPChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockFlusher_UnregisterPChannel_Call) Return() *MockFlusher_UnregisterPChannel_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFlusher_UnregisterPChannel_Call) RunAndReturn(run func(string)) *MockFlusher_UnregisterPChannel_Call {
	_c.Call.Return(run)
	return _c
}

// UnregisterVChannel provides a mock function with given fields: vchannel
func (_m *MockFlusher) UnregisterVChannel(vchannel string) {
	_m.Called(vchannel)
}

// MockFlusher_UnregisterVChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UnregisterVChannel'
type MockFlusher_UnregisterVChannel_Call struct {
	*mock.Call
}

// UnregisterVChannel is a helper method to define mock.On call
//   - vchannel string
func (_e *MockFlusher_Expecter) UnregisterVChannel(vchannel interface{}) *MockFlusher_UnregisterVChannel_Call {
	return &MockFlusher_UnregisterVChannel_Call{Call: _e.mock.On("UnregisterVChannel", vchannel)}
}

func (_c *MockFlusher_UnregisterVChannel_Call) Run(run func(vchannel string)) *MockFlusher_UnregisterVChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockFlusher_UnregisterVChannel_Call) Return() *MockFlusher_UnregisterVChannel_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockFlusher_UnregisterVChannel_Call) RunAndReturn(run func(string)) *MockFlusher_UnregisterVChannel_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockFlusher creates a new instance of MockFlusher. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockFlusher(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockFlusher {
	mock := &MockFlusher{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
