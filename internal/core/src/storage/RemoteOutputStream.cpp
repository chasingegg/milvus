#include "RemoteOutputStream.h"
#include <cstddef>
#include "common/EasyAssert.h"

namespace milvus::storage {

RemoteOutputStream::RemoteOutputStream(std::shared_ptr<arrow::io::OutputStream>&& output_stream)
    : output_stream_(std::move(output_stream)) {
}

size_t
RemoteOutputStream::Tell() const {
    auto status = output_stream_->Tell();
    AssertInfo(status.ok(), "Failed to tell output stream");
    return status.ValueOrDie();
}

size_t
RemoteOutputStream::Write(const void* data, size_t size) {
    auto status = output_stream_->Write(data, size);
    AssertInfo(status.ok(), "Failed to write to output stream");
    return size;
}

}  // namespace milvus::storage