#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace parser {

class MappedFile {
public:
    MappedFile() = default;

    explicit MappedFile(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            int failure = errno;
            close(fd);
            throw std::runtime_error("cannot stat " + path + ": " + std::strerror(failure));
        }
        if (st.st_size <= 0) {
            close(fd);
            throw std::runtime_error("empty file: " + path);
        }

        size_t size = static_cast<size_t>(st.st_size);
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        int failure = errno;
        close(fd);

        if (mapped == MAP_FAILED) {
            throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(failure));
        }

        data_ = static_cast<const char*>(mapped);
        size_ = size;
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            Unmap();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~MappedFile() { Unmap(); }

    void Advise(int advice) const {
        if (data_ != nullptr) {
            madvise(const_cast<char*>(data_), size_, advice);
        }
    }

    const char* data() const { return data_; }
    size_t size() const { return size_; }
    std::string_view view() const { return {data_, size_}; }

private:
    void Unmap() {
        if (data_ != nullptr) {
            munmap(const_cast<char*>(data_), size_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    const char* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace parser
