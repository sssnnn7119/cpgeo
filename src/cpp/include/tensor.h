#pragma once

#include <vector>
#include <span>
#include <optional>

namespace bspmap {
class Tensor{
    protected:
        std::optional<std::vector<double>> owned_data_;  // 只在拥有数据时使用
        std::span<double> data_view_;  // 总是用这个访问数据
        std::vector<int> shape_;
        int dimensions_;
    public:
        Tensor(std::span<const int> shape);
        Tensor(std::span<const int> shape, std::span<double> data);
        double& operator[](int index);
        double& operator[](std::vector<int> indices);
        const double& operator[](int index) const;
        const double& operator[](std::vector<int> indices) const;
        const size_t size() const;
        const double& get(std::vector<int> indices) const;
        const std::vector<int>& get_shape() const { return shape_; }
        int get_dimensions() const { return dimensions_; }
};

class Tensor2D : public Tensor {
    public:
        Tensor2D(int rows, int cols);
        Tensor2D(int rows, int cols, std::span<double> data);
        double& at(int row, int col);
        const double& at(int row, int col) const;

        // get a specific row or column as Tensor1D ([row][col])
        std::span<double> row(int index);
};

class Tensor3D : public Tensor {
    public:
        Tensor3D(int dim1, int dim2, int dim3);
        Tensor3D(int dim1, int dim2, int dim3, std::span<double> data);
        double& at(int i, int j, int k);
        const double& at(int i, int j, int k) const;
        // get a specific slice as span of spans
        std::vector<std::span<double>> slice(int index);
};


} // namespace bspmap