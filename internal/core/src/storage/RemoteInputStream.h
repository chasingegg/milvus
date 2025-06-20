#pragma once

#include "filemanager/InputStream.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus::storage {

class RemoteInputStream : public milvus::InputStream {
 public:
    explicit RemoteInputStream(std::shared_ptr<arrow::io::RandomAccessFile>&& remote_file);

    ~RemoteInputStream() override = default;

    size_t
    Size() const override;

    size_t
    Read(void* data, size_t size) override;

    size_t
    ReadAt(void* data, size_t offset, size_t size) override;

    size_t
    Tell() const override;

    bool
    Eof() const override;

    bool
    Seek(int64_t offset) override;

 private:
   size_t file_size_;
   std::shared_ptr<arrow::io::RandomAccessFile> remote_file_;
};

}  // namespace milvus::storage