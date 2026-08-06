#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace model {

constexpr double PI = 3.14159265358979323846;

class Matrix {
public:
    Matrix() : rows_(0), cols_(0), data_(0) {}

    Matrix(size_t rows, size_t cols, float value = 0.0f)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

    Matrix& operator=(const Matrix& other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        return *this;
    }

    Matrix(Matrix&& other) : rows_(std::exchange(other.rows_, 0)),
                             cols_(std::exchange(other.cols_, 0)),
                             data_(std::move(other.data_)) {}

    Matrix& operator=(Matrix&& other) {
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        data_ = std::move(other.data_);
        return *this;
    }

    float& operator[](size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& operator[](size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    void allocate_rows(size_t rows) {
        data_.resize((rows_ + rows) * cols_);
        rows_ += rows;
    }

    void add_row(const std::vector<float> row) {
        data_.reserve((rows_ + 1) * cols_);
        rows_++;

        for (const auto& elem : row) {
            data_.push_back(elem);
        }
    }

    void add_row(float* row, size_t row_size) {
        data_.reserve((rows_ + 1) * cols_);
        rows_++;

        for (size_t i = 0; i < row_size; ++i) {
            data_.push_back(row[i]);
        }
    }

    float* row_ptr(size_t row) {
        return data_.data() + row * cols_;
    }
    const float* row_ptr(size_t row) const {
        return data_.data() + row * cols_;
    }

    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }

    Matrix last_row() const {
        Matrix row{1, cols_};
        for (size_t i = 0; i < cols_; ++i) {
            row[0, i] = data_[(rows_ - 1) * cols_ + i];
        }
        return row;
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
};

}  // namespace model
