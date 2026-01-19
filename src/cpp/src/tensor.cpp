
#include <vector>
#include <array>
#include "tensor.h"
#include <span>
#include <optional>

namespace bspmap {

Tensor::Tensor(std::span<const int> shape)
    : shape_(shape.begin(), shape.end()) 
{
    this->dimensions_ = static_cast<int>(shape.size());

    int total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }

    // 创建自己的数据并初始化为 0
    owned_data_ = std::vector<double>(static_cast<size_t>(total_size), 0.0);
    data_view_ = std::span<double>(owned_data_->data(), owned_data_->size());
}

Tensor::Tensor(std::span<const int> shape, std::span<double> data)
    : shape_(shape.begin(), shape.end()), data_view_(data)
{
    this->dimensions_ = static_cast<int>(shape.size());
    // 不拥有数据，只使用外部 span
}

double& Tensor::operator[](int index) {
    return data_view_[index];
}

double& Tensor::operator[](std::vector<int> indices) {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_view_[index];
}

const double& Tensor::operator[](int index) const {
    return data_view_[index];
}

const double& Tensor::operator[](std::vector<int> indices) const {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_view_[index];
}

const double& Tensor::get(std::vector<int> indices) const {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_view_[index];
}

const size_t Tensor::size() const {
    return data_view_.size();
}


Tensor2D::Tensor2D(int rows, int cols)
    : Tensor(std::array<int, 2>{rows, cols}) {}

Tensor2D::Tensor2D(int rows, int cols, std::span<double> data)
    : Tensor(std::array<int, 2>{rows, cols}, data) {}

double& Tensor2D::at(int row, int col) {
    return this->operator[]({row, col});
}

const double& Tensor2D::at(int row, int col) const {
    return this->operator[]({row, col});
}

std::span<double> Tensor2D::row(int index) {
    auto shape = get_shape();
    int cols = shape[1];
    return std::span<double>(&data_view_[static_cast<size_t>(index * cols)], static_cast<size_t>(cols));
}

Tensor3D::Tensor3D(int dim1, int dim2, int dim3)
    : Tensor(std::array<int, 3>{dim1, dim2, dim3}) {}

Tensor3D::Tensor3D(int dim1, int dim2, int dim3, std::span<double> data)
    : Tensor(std::array<int, 3>{dim1, dim2, dim3}, data) {}

double& Tensor3D::at(int i, int j, int k) {
    return this->operator[]({i, j, k});
}

const double& Tensor3D::at(int i, int j, int k) const {
    return this->operator[]({i, j, k});
}

std::vector<std::span<double>> Tensor3D::slice(int index) {
    std::vector<std::span<double>> slices;
    auto shape = get_shape();
    int dim2 = shape[1];
    int dim3 = shape[2];
    slices.reserve(static_cast<size_t>(dim2));
    for (int j = 0; j < dim2; ++j) {
        std::span<double> span_now = std::span<double>(&data_view_[static_cast<size_t>(index * dim2 * dim3) + static_cast<size_t>(j * dim3)], static_cast<size_t>(dim3));
        slices.push_back(span_now);
    }
    
    return slices;
}


}