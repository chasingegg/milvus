#include "RemoteInputStream.h"
#include "common/EasyAssert.h"
#include "arrow/buffer.h"
#include "arrow/util/future.h"
#include "arrow/result.h"


namespace milvus::storage {

using ::arrow::Buffer;
using ::arrow::Future;

RemoteInputStream::RemoteInputStream(std::shared_ptr<arrow::io::RandomAccessFile>&& remote_file)
    : remote_file_(std::move(remote_file)) {
    auto status = remote_file_->GetSize();
    AssertInfo(status.ok(), "Failed to get size of remote file");
    file_size_ = static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::Read(void* data, size_t size) {
    auto status = remote_file_->Read(size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadAt(void* data, size_t offset, size_t size) {
    auto status = remote_file_->ReadAt(offset, size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadAtAsync(std::vector<void*>& data, const std::vector<size_t>& offset, const std::vector<size_t>& size) {
    std::vector<Future<std::shared_ptr<Buffer>>> futures;
    futures.reserve(offset.size());
    for (size_t i = 0; i < offset.size(); ++i) {
        futures.emplace_back(remote_file_->ReadAsync(offset[i], size[i]));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
        auto buf = (*futures[i].result());
        std::memcpy(data[i], buf->data(), size[i]);
    }
    return 0;
    // auto status = remote_file_->ReadAt(offset, size, data);
    // AssertInfo(status.ok(), "Failed to read from input stream");
    // return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::Tell() const {
    auto status = remote_file_->Tell();
    AssertInfo(status.ok(), "Failed to tell input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

bool
RemoteInputStream::Eof() const {
    return Tell() >= file_size_;
}

bool
RemoteInputStream::Seek(int64_t offset) {
    auto status = remote_file_->Seek(offset);
    return status.ok();
}

size_t
RemoteInputStream::Size() const {
    return file_size_;
}

}  // namespace milvus::storage