#include "RemoteInputStream.h"
#include "common/EasyAssert.h"
#include "arrow/buffer.h"
#include "arrow/util/future.h"
#include "arrow/result.h"


namespace milvus::storage {

using ::arrow::Buffer;
using ::arrow::Future;

RemoteInputStream::RemoteInputStream(std::string bucket, std::string file_key, std::shared_ptr<milvus_storage::S3CrtClientWrapper> client)
    : bucket_(std::move(bucket)),
      file_key_(std::move(file_key)),
      client_(client) {
    file_size_ = client_->GetObjectSize(bucket_, file_key_);
}

size_t
RemoteInputStream::Read(void* data, size_t size) {
    return client_->GetObjectRange(bucket_, file_key_, pos_, pos_ + size, data);
    // auto status = remote_file_->Read(size, data);
    // AssertInfo(status.ok(), "Failed to read from input stream");
    // return static_cast<size_t>(status.ValueOrDie());
    
}

size_t
RemoteInputStream::ReadAt(void* data, size_t offset, size_t size) {
    return client_->GetObjectRange(bucket_, file_key_, offset, offset + size, data);
    // auto status = remote_file_->ReadAt(offset, size, data);
    // AssertInfo(status.ok(), "Failed to read from input stream");
    // return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadAtAsync(std::vector<void*>& data, const std::vector<size_t>& offset, const std::vector<size_t>& size) {
    // std::vector<Future<std::shared_ptr<Buffer>>> futures;
    // futures.reserve(offset.size());
    // for (size_t i = 0; i < offset.size(); ++i) {
    //     futures.emplace_back(remote_file_->ReadAsync(offset[i], size[i]));
    // }
    // for (size_t i = 0; i < futures.size(); ++i) {
    //     auto buf = (*futures[i].result());
    //     std::memcpy(data[i], buf->data(), size[i]);
    // }
    return 0;
    // auto status = remote_file_->ReadAt(offset, size, data);
    // AssertInfo(status.ok(), "Failed to read from input stream");
    // return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadToFileAsync(const std::vector<size_t>& offset, const std::vector<size_t>& size,
    const std::string& local_file_path, const std::vector<int64_t>& ids, const std::function<void(int)>& callback) {
        
    return 0;
}

size_t
RemoteInputStream::Tell() const {
    return pos_;
    // auto status = remote_file_->Tell();
    // AssertInfo(status.ok(), "Failed to tell input stream");
    // return static_cast<size_t>(status.ValueOrDie());
}

bool
RemoteInputStream::Eof() const {
    return Tell() >= file_size_;
}

bool
RemoteInputStream::Seek(int64_t offset) {
    // auto status = remote_file_->Seek(offset);
    // return status.ok();
    pos_ = offset;
    return true;
}

size_t
RemoteInputStream::Size() const {
    return file_size_;
}

}  // namespace milvus::storage