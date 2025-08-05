#pragma once

#include "filemanager/OutputStream.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus::storage {

class RemoteOutputStream : public milvus::OutputStream {
 public:
    explicit RemoteOutputStream(std::shared_ptr<arrow::io::OutputStream>&& output_stream);

    ~RemoteOutputStream() override = default;

    size_t
    Tell() const override;

    size_t
    Write(const void* data, size_t size) override;


 private:
    std::shared_ptr<arrow::io::OutputStream> output_stream_;
};

}  // namespace milvus::storage