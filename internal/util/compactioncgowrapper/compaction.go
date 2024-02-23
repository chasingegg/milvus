package compactioncgowrapper

/*
#cgo pkg-config: milvus_indexbuilder

#include <stdlib.h>	// free
#include "indexbuilder/index_c.h"
*/
import "C"

import (
	"context"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/golang/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/indexcgopb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
)

type Blob = storage.Blob

type IndexFileInfo struct {
	FileName string
	FileSize int64
}

type CodecIndex interface {
	Serialize() ([]*Blob, error)
	GetIndexFileInfo() ([]*IndexFileInfo, error)
	Load([]*Blob) error
	Delete() error
	CleanLocalData() error
	UpLoad() (map[string]int64, error)
	UpLoadV2() (int64, error)
}

var _ CodecIndex = (*CgoIndex)(nil)

type CgoIndex struct {
	indexPtr C.CIndex
	close    bool
}

// TODO: use proto.Marshal instead of proto.MarshalTextString for better compatibility.
func NewCgoIndex(dtype schemapb.DataType, typeParams, indexParams map[string]string) (CodecIndex, error) {
	protoTypeParams := &indexcgopb.TypeParams{
		Params: make([]*commonpb.KeyValuePair, 0),
	}
	for key, value := range typeParams {
		protoTypeParams.Params = append(protoTypeParams.Params, &commonpb.KeyValuePair{Key: key, Value: value})
	}
	typeParamsStr := proto.MarshalTextString(protoTypeParams)

	protoIndexParams := &indexcgopb.IndexParams{
		Params: make([]*commonpb.KeyValuePair, 0),
	}
	for key, value := range indexParams {
		protoIndexParams.Params = append(protoIndexParams.Params, &commonpb.KeyValuePair{Key: key, Value: value})
	}
	indexParamsStr := proto.MarshalTextString(protoIndexParams)

	typeParamsPointer := C.CString(typeParamsStr)
	indexParamsPointer := C.CString(indexParamsStr)
	defer C.free(unsafe.Pointer(typeParamsPointer))
	defer C.free(unsafe.Pointer(indexParamsPointer))

	var indexPtr C.CIndex
	cintDType := uint32(dtype)
	status := C.CreateIndexV0(cintDType, typeParamsPointer, indexParamsPointer, &indexPtr)
	if err := HandleCStatus(&status, "failed to create index"); err != nil {
		return nil, err
	}

	index := &CgoIndex{
		indexPtr: indexPtr,
		close:    false,
	}

	runtime.SetFinalizer(index, func(index *CgoIndex) {
		if index != nil && !index.close {
			log.Error("there is leakage in index object, please check.")
		}
	})

	return index, nil
}

func CreateIndex(ctx context.Context, buildIndexInfo *BuildIndexInfo) (CodecIndex, error) {
	var indexPtr C.CIndex
	status := C.CreateIndex(&indexPtr, buildIndexInfo.cBuildIndexInfo)
	if err := HandleCStatus(&status, "failed to create index"); err != nil {
		return nil, err
	}

	index := &CgoIndex{
		indexPtr: indexPtr,
		close:    false,
	}

	return index, nil
}

func CreateIndexV2(ctx context.Context, buildIndexInfo *BuildIndexInfo) (CodecIndex, error) {
	var indexPtr C.CIndex
	status := C.CreateIndexV2(&indexPtr, buildIndexInfo.cBuildIndexInfo)
	if err := HandleCStatus(&status, "failed to create index"); err != nil {
		return nil, err
	}

	index := &CgoIndex{
		indexPtr: indexPtr,
		close:    false,
	}

	return index, nil
}

func (index *CgoIndex) Serialize() ([]*Blob, error) {
	var cBinarySet C.CBinarySet

	status := C.SerializeIndexToBinarySet(index.indexPtr, &cBinarySet)
	defer func() {
		if cBinarySet != nil {
			C.DeleteBinarySet(cBinarySet)
		}
	}()
	if err := HandleCStatus(&status, "failed to serialize index to binary set"); err != nil {
		return nil, err
	}

	keys, err := GetBinarySetKeys(cBinarySet)
	if err != nil {
		return nil, err
	}
	ret := make([]*Blob, 0)
	for _, key := range keys {
		value, err := GetBinarySetValue(cBinarySet, key)
		if err != nil {
			return nil, err
		}
		size, err := GetBinarySetSize(cBinarySet, key)
		if err != nil {
			return nil, err
		}
		blob := &Blob{
			Key:   key,
			Value: value,
			Size:  size,
		}
		ret = append(ret, blob)
	}

	return ret, nil
}

func (index *CgoIndex) GetIndexFileInfo() ([]*IndexFileInfo, error) {
	var cBinarySet C.CBinarySet

	status := C.SerializeIndexToBinarySet(index.indexPtr, &cBinarySet)
	defer func() {
		if cBinarySet != nil {
			C.DeleteBinarySet(cBinarySet)
		}
	}()
	if err := HandleCStatus(&status, "failed to serialize index to binary set"); err != nil {
		return nil, err
	}

	keys, err := GetBinarySetKeys(cBinarySet)
	if err != nil {
		return nil, err
	}
	ret := make([]*IndexFileInfo, 0)
	for _, key := range keys {
		size, err := GetBinarySetSize(cBinarySet, key)
		if err != nil {
			return nil, err
		}
		info := &IndexFileInfo{
			FileName: key,
			FileSize: size,
		}
		ret = append(ret, info)
	}

	return ret, nil
}

func (index *CgoIndex) Load(blobs []*Blob) error {
	var cBinarySet C.CBinarySet
	status := C.NewBinarySet(&cBinarySet)
	defer C.DeleteBinarySet(cBinarySet)

	if err := HandleCStatus(&status, "failed to load index"); err != nil {
		return err
	}
	for _, blob := range blobs {
		key := blob.Key
		byteIndex := blob.Value
		indexPtr := unsafe.Pointer(&byteIndex[0])
		indexLen := C.int64_t(len(byteIndex))
		binarySetKey := filepath.Base(key)
		indexKey := C.CString(binarySetKey)
		status = C.AppendIndexBinary(cBinarySet, indexPtr, indexLen, indexKey)
		C.free(unsafe.Pointer(indexKey))
		if err := HandleCStatus(&status, "failed to load index"); err != nil {
			return err
		}
	}
	status = C.LoadIndexFromBinarySet(index.indexPtr, cBinarySet)
	return HandleCStatus(&status, "failed to load index")
}

func (index *CgoIndex) Delete() error {
	if index.close {
		return nil
	}
	status := C.DeleteIndex(index.indexPtr)
	index.close = true
	return HandleCStatus(&status, "failed to delete index")
}

func (index *CgoIndex) CleanLocalData() error {
	status := C.CleanLocalData(index.indexPtr)
	return HandleCStatus(&status, "failed to clean cached data on disk")
}

func (index *CgoIndex) UpLoad() (map[string]int64, error) {
	var cBinarySet C.CBinarySet

	status := C.SerializeIndexAndUpLoad(index.indexPtr, &cBinarySet)
	defer func() {
		if cBinarySet != nil {
			C.DeleteBinarySet(cBinarySet)
		}
	}()
	if err := HandleCStatus(&status, "failed to serialize index and upload index"); err != nil {
		return nil, err
	}

	res := make(map[string]int64)
	indexFilePaths, err := GetBinarySetKeys(cBinarySet)
	if err != nil {
		return nil, err
	}
	for _, path := range indexFilePaths {
		size, err := GetBinarySetSize(cBinarySet, path)
		if err != nil {
			return nil, err
		}
		res[path] = size
	}

	runtime.SetFinalizer(index, func(index *CgoIndex) {
		if index != nil && !index.close {
			log.Error("there is leakage in index object, please check.")
		}
	})

	return res, nil
}

func (index *CgoIndex) UpLoadV2() (int64, error) {
	var cBinarySet C.CBinarySet

	status := C.SerializeIndexAndUpLoadV2(index.indexPtr, &cBinarySet)
	defer func() {
		if cBinarySet != nil {
			C.DeleteBinarySet(cBinarySet)
		}
	}()
	if err := HandleCStatus(&status, "failed to serialize index and upload index"); err != nil {
		return -1, err
	}

	buffer, err := GetBinarySetValue(cBinarySet, "index_store_version")
	if err != nil {
		return -1, err
	}
	var version int64

	version = int64(buffer[7])
	version = (version << 8) + int64(buffer[6])
	version = (version << 8) + int64(buffer[5])
	version = (version << 8) + int64(buffer[4])
	version = (version << 8) + int64(buffer[3])
	version = (version << 8) + int64(buffer[2])
	version = (version << 8) + int64(buffer[1])
	version = (version << 8) + int64(buffer[0])

	runtime.SetFinalizer(index, func(index *CgoIndex) {
		if index != nil && !index.close {
			log.Error("there is leakage in index object, please check.")
		}
	})

	return version, nil
}
