#pragma once

#include "filemanager/InputStream.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus::storage {

class RemoteInputStream : public milvus::InputStream {
 public:
    explicit RemoteInputStream(std::string bucket, std::string file_key, std::shared_ptr<milvus_storage::S3CrtClientWrapper> client,
      std::shared_ptr<arrow::io::RandomAccessFile> remote_file);

    ~RemoteInputStream() override = default;

    size_t
    Size() const override;

    size_t
    Read(void* data, size_t size) override;

    size_t
    ReadAt(void* data, size_t offset, size_t size) override;

    size_t
    ReadAtAsync(std::vector<void*>& data, const std::vector<size_t>& offset, const std::vector<size_t>& size) override;

    size_t
    ReadToFileAsync(const std::vector<size_t>& offset, const std::vector<size_t>& size,
      const std::string& local_file_path, const std::vector<int64_t>& ids, const std::function<void(int)>& callback) override;

    size_t
    Tell() const override;

    bool
    Eof() const override;

    bool
    Seek(int64_t offset) override;

 private:
   size_t pos_ = 0;
   size_t file_size_;
   std::string bucket_;
   std::string file_key_;
   std::shared_ptr<arrow::io::RandomAccessFile> remote_file_;
   std::shared_ptr<milvus_storage::S3CrtClientWrapper> client_;
};

}  // namespace milvus::storage