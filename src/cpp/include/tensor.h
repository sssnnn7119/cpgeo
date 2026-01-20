


#pragma once

#include <vector>
#include <span>
#include <memory>

namespace cpgeo {

// ========== TensorView: 引用外部数据 ==========
class TensorView {
    protected:
        std::span<double> data_;  // 引用外部数据
        std::vector<int> shape_;
        int dimensions_;

    public:
        // 从外部 span 创建视图
        TensorView(std::span<const int> shape, std::span<double> data);
        
        // 访问器
        double& operator[](int index);
        double& operator[](std::vector<int> indices);
        const double& operator[](int index) const;
        const double& operator[](std::vector<int> indices) const;
        
        const size_t size() const { return data_.size(); }
        const double& get(std::vector<int> indices) const;
        const std::vector<int>& get_shape() const { return shape_; }
        int get_dimensions() const { return dimensions_; }
        std::span<double> data() { return data_; }
        std::span<const double> data() const { return data_; }
};

// ========== Tensor: 拥有自己的数据 ==========
class Tensor {
    protected:
        std::vector<double> data_;  // 拥有的数据
        std::vector<int> shape_;
        int dimensions_;

    public:
        // 创建并拥有数据（初始化为 0）
        Tensor(std::span<const int> shape);
        
        // 从外部数据拷贝创建
        Tensor(std::span<const int> shape, std::span<const double> data);
        
        // 移动构造和赋值（正确处理数据所有权）
        Tensor(Tensor&& other) noexcept = default;
        Tensor& operator=(Tensor&& other) noexcept = default;
        
        // 禁用拷贝（如果需要拷贝，显式实现）
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        
        // 访问器
        double& operator[](int index);
        double& operator[](std::vector<int> indices);
        const double& operator[](int index) const;
        const double& operator[](std::vector<int> indices) const;
        
        const size_t size() const { return data_.size(); }
        const double& get(std::vector<int> indices) const;
        const std::vector<int>& get_shape() const { return shape_; }
        int get_dimensions() const { return dimensions_; }
        
        // 获取数据指针
        double* data_ptr() { return data_.data(); }
        const double* data_ptr() const { return data_.data(); }
        
        // 创建视图（不转移所有权）
        TensorView view() { return TensorView(shape_, std::span<double>(data_)); }
};

// ========== Tensor2D: 拥有数据的 2D 张量 ==========
class Tensor2D : public Tensor {
    public:
        Tensor2D(int rows, int cols);
        Tensor2D(int rows, int cols, std::span<const double> data);
        
        double& at(int row, int col);
        const double& at(int row, int col) const;
        std::span<double> row(int index);
};

// ========== Tensor3D: 拥有数据的 3D 张量 ==========
class Tensor3D : public Tensor {
    public:
        Tensor3D(int dim1, int dim2, int dim3);
        Tensor3D(int dim1, int dim2, int dim3, std::span<const double> data);
        
        double& at(int i, int j, int k);
        const double& at(int i, int j, int k) const;
        std::vector<std::span<double>> slice(int index);
};

// ========== TensorView2D: 2D 张量视图 ==========
class TensorView2D : public TensorView {
    public:
        TensorView2D(int rows, int cols, std::span<double> data);
        
        double& at(int row, int col);
        const double& at(int row, int col) const;
        std::span<double> row(int index);
};

// ========== TensorView3D: 3D 张量视图 ==========
class TensorView3D : public TensorView {
    public:
        TensorView3D(int dim1, int dim2, int dim3, std::span<double> data);
        
        double& at(int i, int j, int k);
        const double& at(int i, int j, int k) const;
        std::vector<std::span<double>> slice(int index);
};

} // namespace bspmap