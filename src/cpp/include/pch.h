// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"

#endif //PCH_H

#ifdef IMPORT_DLL
#else
#define IMPORT_DLL extern "C" _declspec(dllimport) //指的是允许将其给外部调用
#endif

// calculate the indices of the points
IMPORT_DLL void* build_trees(double* knots, double* threshold, int num_knots);
IMPORT_DLL void delete_trees(void* base_tree);
IMPORT_DLL void cal_indices(void* base_tree, double* points, int num_pts, int* sizes);
IMPORT_DLL void get_indices(int* results);

// calculate the 2nd indices
IMPORT_DLL void cal_indices_2nd(const int* indices, int n, int* indices_2_size);
IMPORT_DLL void get_indices_2nd(int* results);

// calculate the threshold
IMPORT_DLL void get_thresholds(double* results, const double* knots, int n, int k);




IMPORT_DLL void refine_triangular_mesh(
	int* result, int* elements,
	int num_elements,
	int* edges,
	int num_edges,
	int* adjacents,
	double* points,
	int num_points);


IMPORT_DLL void triangular_mesh(int* num_mesh, double* nodes, int num_nodes);
IMPORT_DLL void get_triangular_mesh(int* results);