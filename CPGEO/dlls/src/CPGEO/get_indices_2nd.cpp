#include "pch.h"
#include <unordered_map>
#include <vector>


using namespace std;


/**
 * 获取indices中第二行相等的列对索引
 *
 * @param indices 输入的2行n列数组，以一维数组形式存储
 * @param n indices的列数
 * @param indices_2_size 输出结果的大小，将被修改
 * @return indices_2数组，2行m列，每列存储两个indices中列的索引，这两列的第二行元素相等
 */
vector<int> results_1, results_2;
void cal_indices_2nd(const int* indices, int n, int* indices_2_size) {

	results_1.clear();
	results_2.clear();



    // 预分配哈希表，减少动态扩容
    std::unordered_map<int, std::vector<int>> value_to_columns;
    value_to_columns.reserve(n / 2 + 1); // 假设约一半的值可能重复

    // 先计算结果大小
    int result_count = 0;

    // 第一次遍历：构建哈希表并计算结果大小
    for (int col = 0; col < n; ++col) {
        int second_row_value = indices[n + col]; // 获取第二行的值
        auto& columns = value_to_columns[second_row_value];

        // 计算与现有列形成的新对数
        result_count += columns.size();

        // 存储当前列索引
        columns.push_back(col);
    }



    // 如果没有结果，返回nullptr
    if (result_count == 0) {
        return;
    }

    // 直接分配输出数组内存
    results_1.reserve(result_count);
    results_2.reserve(result_count);

    // 填充输出数组
    int pair_index = 0;

    // 第二次遍历：填充结果数组
    for (const auto& entry : value_to_columns) {
        const auto& columns = entry.second;
        if (columns.size() > 1) {
            for (size_t i = 0; i < columns.size(); ++i) {
                for (size_t j = i + 1; j < columns.size(); ++j) {
                    // 直接填充到对应位置
                    results_1.push_back(columns[i]);
                    results_2.push_back(columns[j]);
                    pair_index++;
                }
            }
        }
    }

    // 设置输出大小
    *indices_2_size = pair_index;
}

void get_indices_2nd(int* results) {
	for (int i = 0; i < results_1.size(); i++) {
		results[i] = results_1[i];
		results[i + results_1.size()] = results_2[i];
	}
}