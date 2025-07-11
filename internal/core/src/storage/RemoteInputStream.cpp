#include "RemoteInputStream.h"
#include "common/EasyAssert.h"
#include "arrow/buffer.h"
#include "arrow/util/future.h"
#include "arrow/result.h"
#include <iostream>


namespace milvus::storage {

using ::arrow::Buffer;
using ::arrow::Future;

RemoteInputStream::RemoteInputStream(std::string bucket, std::string file_key, std::shared_ptr<milvus_storage::S3CrtClientWrapper> client)
    : bucket_(std::move(bucket)),
      file_key_(std::move(file_key)),
      client_(client) {
    std::cout << "FUCK want to get size " << bucket_ << " " << file_key_ << std::endl;
    file_size_ = client_->GetObjectSize(bucket_, file_key_);
    std::cout << "FUCK want to get size done " << file_size_ << std::endl;
}

size_t
RemoteInputStream::Read(void* data, size_t size) {
    // return 0;
    std::cout << "FUCK read " << pos_ << " " << size << std::endl;
    client_->GetObjectRange(bucket_, file_key_, pos_, size, data);
    pos_ += size;
    std::cout << "FUCK read done " << pos_ << " " << size << std::endl;
    return size;
}

size_t
RemoteInputStream::ReadAt(void* data, size_t offset, size_t size) {
    // return 0;
    return client_->GetObjectRange(bucket_, file_key_, offset, size, data);
}

size_t
RemoteInputStream::ReadAtAsync(std::vector<void*>& data, const std::vector<size_t>& offset, const std::vector<size_t>& size) {
    // return 0;
    return client_->ReadBatchToMemory(bucket_, file_key_, data, offset, size);
}

size_t
RemoteInputStream::ReadToFileAsync(const std::vector<size_t>& offset, const std::vector<size_t>& size,
    const std::string& local_file_path, const std::vector<int64_t>& ids, const std::function<void(int)>& callback) {
    // return 0;
    return client_->ReadBatchToFile(bucket_, file_key_, local_file_path, offset, size, ids, callback);
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