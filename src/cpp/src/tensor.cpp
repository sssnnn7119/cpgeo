
#include <vector>
#include <array>
#include "tensor.h"
#include <span>
#include <algorithm>

namespace cpgeo {

// ========== TensorView 实现 ==========
TensorView::TensorView(std::span<const int> shape, std::span<double> data)
    : data_(data), shape_(shape.begin(), shape.end()), dimensions_(static_cast<int>(shape.size()))
{}

double& TensorView::operator[](int index) {
    return data_[index];
}

double& TensorView::operator[](std::vector<int> indices) {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_[index];
}

const double& TensorView::operator[](int index) const {
    return data_[index];
}

const double& TensorView::operator[](std::vector<int> indices) const {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_[index];
}

const double& TensorView::get(std::vector<int> indices) const {
    return (*this)[indices];
}

// ========== Tensor 实现 ==========
Tensor::Tensor(std::span<const int> shape)
    : shape_(shape.begin(), shape.end()), dimensions_(static_cast<int>(shape.size()))
{
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= static_cast<size_t>(dim);
    }
    data_.resize(total_size, 0.0);
}

Tensor::Tensor(std::span<const int> shape, std::span<const double> data)
    : shape_(shape.begin(), shape.end()), dimensions_(static_cast<int>(shape.size()))
{
    data_.assign(data.begin(), data.end());
}

double& Tensor::operator[](int index) {
    return data_[index];
}

double& Tensor::operator[](std::vector<int> indices) {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_[index];
}

const double& Tensor::operator[](int index) const {
    return data_[index];
}

const double& Tensor::operator[](std::vector<int> indices) const {
    int index = 0;
    for (int d = 0; d < dimensions_; ++d) {
        index *= shape_[d];
        index += indices[d];
    }
    return data_[index];
}

const double& Tensor::get(std::vector<int> indices) const {
    return (*this)[indices];
}

// ========== Tensor2D 实现 ==========
Tensor2D::Tensor2D(int rows, int cols)
    : Tensor(std::array<int, 2>{rows, cols}) {}

Tensor2D::Tensor2D(int rows, int cols, std::span<const double> data)
    : Tensor(std::array<int, 2>{rows, cols}, data) {}

double& Tensor2D::at(int row, int col) {
    return this->operator[]({row, col});
}

const double& Tensor2D::at(int row, int col) const {
    return this->operator[]({row, col});
}

std::span<double> Tensor2D::row(int index) {
    int cols = shape_[1];
    return std::span<double>(&data_[static_cast<size_t>(index * cols)], static_cast<size_t>(cols));
}

// ========== Tensor3D 实现 ==========
Tensor3D::Tensor3D(int dim1, int dim2, int dim3)
    : Tensor(std::array<int, 3>{dim1, dim2, dim3}) {}

Tensor3D::Tensor3D(int dim1, int dim2, int dim3, std::span<const double> data)
    : Tensor(std::array<int, 3>{dim1, dim2, dim3}, data) {}

double& Tensor3D::at(int i, int j, int k) {
    return this->operator[]({i, j, k});
}

const double& Tensor3D::at(int i, int j, int k) const {
    return this->operator[]({i, j, k});
}

std::vector<std::span<double>> Tensor3D::slice(int index) {
    std::vector<std::span<double>> slices;
    int dim2 = shape_[1];
    int dim3 = shape_[2];
    slices.reserve(static_cast<size_t>(dim2));
    
    for (int j = 0; j < dim2; ++j) {
        size_t offset = static_cast<size_t>(index * dim2 * dim3 + j * dim3);
        slices.push_back(std::span<double>(&data_[offset], static_cast<size_t>(dim3)));
    }
    
    return slices;
}

// ========== TensorView2D 实现 ==========
TensorView2D::TensorView2D(int rows, int cols, std::span<double> data)
    : TensorView(std::array<int, 2>{rows, cols}, data) {}

double& TensorView2D::at(int row, int col) {
    return this->operator[]({row, col});
}

const double& TensorView2D::at(int row, int col) const {
    return this->operator[]({row, col});
}

std::span<double> TensorView2D::row(int index) {
    int cols = shape_[1];
    return std::span<double>(&data_[static_cast<size_t>(index * cols)], static_cast<size_t>(cols));
}

// ========== TensorView3D 实现 ==========
TensorView3D::TensorView3D(int dim1, int dim2, int dim3, std::span<double> data)
    : TensorView(std::array<int, 3>{dim1, dim2, dim3}, data) {}

double& TensorView3D::at(int i, int j, int k) {
    return this->operator[]({i, j, k});
}

const double& TensorView3D::at(int i, int j, int k) const {
    return this->operator[]({i, j, k});
}

std::vector<std::span<double>> TensorView3D::slice(int index) {
    std::vector<std::span<double>> slices;
    int dim2 = shape_[1];
    int dim3 = shape_[2];
    slices.reserve(static_cast<size_t>(dim2));
    
    for (int j = 0; j < dim2; ++j) {
        size_t offset = static_cast<size_t>(index * dim2 * dim3 + j * dim3);
        slices.push_back(std::span<double>(&data_[offset], static_cast<size_t>(dim3)));
    }
    
    return slices;
}

}

