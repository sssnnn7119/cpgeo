#include "pch.h"
#include <math.h>
#include <iostream>
#include <windows.h>
#include <map>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>


using namespace std;
using namespace boost::multiprecision;

typedef cpp_bin_float_50 lf;


template <typename T>
class vector3d
{
public:
	T x, y, z;
	vector3d() : x(0), y(0), z(0) {}
	vector3d(T x, T y, T z) : x(x), y(y), z(z) {}
	vector3d(const vector3d &v) : x(v.x), y(v.y), z(v.z) {}

	T dot(const vector3d &v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}
	vector3d cross(const vector3d &v) const
	{
		return vector3d(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	vector3d operator+(const vector3d &v) const
	{
		return vector3d(x + v.x, y + v.y, z + v.z);
	}
	vector3d operator-(const vector3d &v) const
	{
		return vector3d(x - v.x, y - v.y, z - v.z);
	}
	vector3d operator*(const T &r) const
	{
		return vector3d(x * r, y * r, z * r);
	}
	vector3d operator/(const T &r) const
	{
		return vector3d(x / r, y / r, z / r);
	}
	T operator*(const vector3d &v) const
	{
		return dot(v);
	}
	vector3d operator-() const
	{
		return vector3d(-x, -y, -z);
	}
	T norm() const
	{
		return sqrt(x * x + y * y + z * z);
	}
	vector3d normalize() const
	{
		return *this / norm();
	}
};

typedef vector3d<lf> vec;


bool same_side(vec p1, vec p2, vec a, vec b)
{
	vec p12 = p2 - p1;

	vec cp1 = p12.cross(a - p1);
	vec cp2 = p12.cross(b - p1);

	cp1 = cp1 / cp1.norm();
	cp2 = cp2 / cp2.norm();
	return cp1 * cp2 <= -0.7;
}

bool find_non_delaunay_edge(int edge_point1, int edge_point2, int another_point1, int another_point2, double *points, int* adjacents, map<uint64_t, int> &edge_to_adjacent, int num_points)
{
	if (adjacents[0] == -1 || adjacents[1] == -1)
	{
		return false;
	}
	//vec e1(points + edge_point1 * 3), e2(points + edge_point2 * 3), a1(points + another_point1 * 3), a2(points + another_point2 * 3);
	vec e1(points[edge_point1 * 3], points[edge_point1 * 3 + 1], points[edge_point1 * 3 + 2]);
	vec e2(points[edge_point2 * 3], points[edge_point2 * 3 + 1], points[edge_point2 * 3 + 2]);
	vec a1(points[another_point1 * 3], points[another_point1 * 3 + 1], points[another_point1 * 3 + 2]);
	vec a2(points[another_point2 * 3], points[another_point2 * 3 + 1], points[another_point2 * 3 + 2]);

	vec a12 = a2 - a1;
	vec a1e1 = e1 - a1;
	vec a1e2 = e2 - a1;

	vec cp1 = a12.cross(a1e1);
	vec cp2 = a12.cross(a1e2);

	cp1 = cp1 / cp1.norm();
	cp2 = cp2 / cp2.norm();

	if (another_point1 > another_point2) {
		swap(another_point1, another_point2);
	}

	if (edge_to_adjacent.find((uint64_t)another_point1 * num_points + another_point2) != edge_to_adjacent.end()) 
	{
		return false;
	}
	if (cp1 * cp2 > -0)
	{
		return false;
	}

	vec e12 = e2 - e1;
	vec a2e1 = e1 - a2;

	vec dp1 = e12.cross(a1e1);
	vec dp2 = e12.cross(a2e1);

	dp1 = dp1 / dp1.norm();
	dp2 = dp2 / dp2.norm();

	if (dp1 * dp2 > cp1 * cp2) {
		return true;
	}

	vec amid = (a1 + a2) / 2;
	vec emid = (e1 + e2) / 2;

	lf elength = (e1 - emid).norm();
	lf ema1 = (emid - a1).norm();
	lf ema2 = (emid - a2).norm();

	if (elength < ema1 && elength < ema2)
	{
		return false;
	}

	lf alength = (a1 - amid).norm();
	lf ame1 = (amid - e1).norm();
	lf ame2 = (amid - e2).norm();
	if (alength < ame1 && alength < ame2)
	{
		return true;
	}

	a12 = a12 / a12.norm();
	a1e1 = a1e1 / a1e1.norm();
	a1e2 = a1e2 / a1e2.norm();

	//vec e12 = e2 - e1;
	//vec a2e1 = e1 - a2;
	vec a2e2 = e2 - a2;

	e12 = e12 / e12.norm();
	a2e1 = a2e1 / a2e1.norm();
	a2e2 = a2e2 / a2e2.norm();

	lf cos1 = -e12 * a1e1;
	lf cos2 = -e12 * a2e1;
	lf cos3 = e12 * a1e2;
	lf cos4 = e12 * a2e2;
	lf cos5 = a1e1 * a1e2;
	lf cos6 = a2e1 * a2e2;
	// minimum of cos1-6
	lf min_cos1_6 = min(cos1, min(cos2, min(cos3, min(cos4, min(cos5, cos6)))));

	cos1 = a12 * a1e1;
	cos2 = a12 * a1e2;
	cos3 = -a12 * a2e1;
	cos4 = -a12 * a2e2;
	cos5 = a1e1 * a2e1;
	cos6 = a1e2 * a2e2;

	lf min_cos7_12 = min(cos1, min(cos2, min(cos3, min(cos4, min(cos5, cos6)))));

	if (min_cos1_6 < min_cos7_12)
	{
		return true;
	}


	return false;
}

// flip one non-delaunay edge
void flip_edge(int edge_index,
			   int *edges,
			   map<uint64_t, int> &edge_to_adjacent,
			   int* another_points_index_now,
			   int *adjacents,
			   int *elements,
			   int num_points)
{
	
	int *edge_now = edges + edge_index * 2;
	int *adjacents_now = adjacents + edge_index * 2;
	int *element_now0 = elements + adjacents_now[0] * 3;
	int *element_now1 = elements + adjacents_now[1] * 3;

	int cond = 0;

	int before_flip[2][3] = {{element_now0[0], element_now0[1], element_now0[2]}, {element_now1[0], element_now1[1], element_now1[2]}};


	element_now0[(another_points_index_now[0] + 2) % 3] = element_now1[another_points_index_now[1]];
	element_now1[(another_points_index_now[1] + 2) % 3] = element_now0[another_points_index_now[0]];


	// update edges



	edge_to_adjacent.erase((uint64_t)edge_now[0] * num_points + (uint64_t)edge_now[1]);

	uint64_t edge0[2] = {edge_now[0], edge_now[1]};

	edge_now[0] = element_now1[another_points_index_now[1]] < element_now0[another_points_index_now[0]] ? element_now1[another_points_index_now[1]] : element_now0[another_points_index_now[0]];
	edge_now[1] = element_now1[another_points_index_now[1]] > element_now0[another_points_index_now[0]] ? element_now1[another_points_index_now[1]] : element_now0[another_points_index_now[0]];

	edge_to_adjacent[(uint64_t)edge_now[0] * num_points + (uint64_t)edge_now[1]] = edge_index;

	// update another_points_index
	int another0[2] = {another_points_index_now[0], another_points_index_now[1]};

	another_points_index_now[0] = (another_points_index_now[0] + 1) % 3;
	another_points_index_now[1] = (another_points_index_now[1] + 1) % 3;

	// update other adjacents and another_points_index

	int temp1, temp2;
	for (int i = 0; i < 2; i++)
	{

		temp1 = before_flip[i][(another0[i]) % 3];
		temp2 = before_flip[i][(another0[i] + 2) % 3];

		if (temp1 > temp2)
		{
			int temp = temp1;
			temp1 = temp2;
			temp2 = temp;
		}

		int influence_edge_index0 = edge_to_adjacent[(uint64_t)temp1 * num_points + (uint64_t)temp2];

		int *adjacents_change = adjacents + influence_edge_index0 * 2;

		int* elem[2] = { elements + adjacents_change[0] * 3, elements + adjacents_change[1] * 3 };
	
		int ind_now= -1;
		if (adjacents_change[0] == adjacents_now[i])
		{
			ind_now = 0;
		}
		else if (adjacents_change[1] == adjacents_now[i])
		{
			ind_now = 1;
		}
		else{
			std::cout<<"error: adjacents_change not found"<<endl;
			std::cout<<adjacents_change[0]<<" "<<adjacents_change[1]<<" "<<adjacents_now[0]<<" "<<adjacents_now[1]<<endl;		
			cout << "before_flip: \n";
			cout<<edge0[0]<<" "<<edge0[1]<<endl;
			cout<<another0[0]<<" "<<another0[1]<<endl;
			cout<<adjacents_now[0]<<": "<<before_flip[0][0]<<" "<<before_flip[0][1]<<" "<<before_flip[0][2]<<endl;
			cout<<adjacents_now[1]<<": "<<before_flip[1][0]<<" "<<before_flip[1][1]<<" "<<before_flip[1][2]<<endl;
			cout << "after_flip: \n";
			cout<<edge_now[0]<<" "<<edge_now[1]<<endl;
			cout<<another_points_index_now[0]<<" "<<another_points_index_now[1]<<endl;
			cout<<adjacents_now[0]<<": "<<element_now0[0]<<" "<<element_now0[1]<<" "<<element_now0[2]<<endl;
			cout<<adjacents_now[1]<<": "<<element_now1[0]<<" "<<element_now1[1]<<" "<<element_now1[2]<<endl;
			cout<<adjacents_change[0]<<": "<<elem[0][0]<<" "<<elem[0][1]<<" "<<elem[0][2]<<endl;
			cout<<adjacents_change[1]<<": "<<elem[1][0]<<" "<<elem[1][1]<<" "<<elem[1][2]<<endl;

			cout<< "temp1: "<<temp1<<" temp2: "<<temp2<<endl;
			cout<<"adjacents_index: "<< influence_edge_index0<<endl;
			cout<<"edges: "<<edges[influence_edge_index0 * 2]<<" "<<edges[influence_edge_index0 * 2 + 1]<<endl;
			throw 1;
		}

		adjacents_change[ind_now] = adjacents_now[1 - i];

	}

	//if ((element_now0[0]) == 2059 || element_now0[1] == 2059 || element_now0[2] == 2059 || element_now1[0] == 2059 || element_now1[1] == 2059 || element_now1[2] == 2059 || 
	//	(element_now0[0]) == 744 || element_now0[1] == 744 || element_now0[2] == 744 || element_now1[0] == 744 || element_now1[1] == 744 || element_now1[2] == 744 ||
	//	(element_now0[0]) == 1297 || element_now0[1] == 1297 || element_now0[2] == 1297 || element_now1[0] == 1297 || element_now1[1] == 1297 || element_now1[2] == 1297 ||
	//	(element_now0[0]) == 2612 || element_now0[1] == 2612 || element_now0[2] == 2612 || element_now1[0] == 2612 || element_now1[1] == 2612 || element_now1[2] == 2612) {
	//
	//	cout << "before_flip: \n";
	//	cout << edge0[0] << " " << edge0[1] << endl;
	//	cout << another0[0] << " " << another0[1] << endl;
	//	cout << adjacents_now[0] << ": " << before_flip[0][0] << " " << before_flip[0][1] << " " << before_flip[0][2] << endl;
	//	cout << adjacents_now[1] << ": " << before_flip[1][0] << " " << before_flip[1][1] << " " << before_flip[1][2] << endl;
	//	cout << "after_flip: \n";
	//	cout << edge_now[0] << " " << edge_now[1] << endl;
	//	cout << another_points_index_now[0] << " " << another_points_index_now[1] << endl;
	//	cout << adjacents_now[0] << ": " << element_now0[0] << " " << element_now0[1] << " " << element_now0[2] << endl;
	//	cout << adjacents_now[1] << ": " << element_now1[0] << " " << element_now1[1] << " " << element_now1[2] << endl;
	//	cout << endl << endl;
	//}

}

// flip all non-delaunay edges
void refine_triangular_mesh(
	int *result,
	int *elements,
	int num_elements,
	int *edges,
	int num_edges,
	int *adjacents,
	double *points,
	int num_points)
{

	bool *be_processed = new bool[num_edges]{false};

	// initialize adjacents and another_points_index
	map<uint64_t, int> edge_to_adjacent;
	for (int i = 0; i < num_edges; i++)
	{
		edge_to_adjacent[(uint64_t)edges[i * 2] * num_points + (uint64_t)edges[i * 2 + 1]] = i;
	}

	// find all non-delaunay edges and flip them
	int index = 0;
	int iter = 0;
	while (true)
	{
		if (index >= num_edges)
		{
			break;
		}

		if (be_processed[index])
		{
			index++;
			continue;
		}

		int *edge_now = edges + index * 2;

		int *adjacents_now = adjacents + index * 2;
		int *element_now0 = elements + adjacents_now[0] * 3;
		int *element_now1 = elements + adjacents_now[1] * 3;

		int another_points_index_now[2];
		for (int j = 0; j < 3; j++)
		{
			if (element_now0[j] != edge_now[0] && element_now0[j] != edge_now[1])
			{
				another_points_index_now[0] = j;
				break;
			}
		}
		for (int j = 0; j < 3; j++)
		{
			if (element_now1[j] != edge_now[0] && element_now1[j] != edge_now[1])
			{
				another_points_index_now[1] = j;
				break;
			}
		}

		bool cond = find_non_delaunay_edge(edge_now[0], edge_now[1], element_now0[another_points_index_now[0]], element_now1[another_points_index_now[1]], points, adjacents_now, edge_to_adjacent, num_points);
		if (cond)
		{

			flip_edge(index, edges, edge_to_adjacent, another_points_index_now, adjacents, elements, num_points);

			// printf("now processing %d ||\r", index);

			int edge_refresh = element_now0[another_points_index_now[0]] < edge_now[0] ? 
			element_now0[another_points_index_now[0]] * num_points + edge_now[0] : 
			edge_now[0] * num_points + element_now0[another_points_index_now[0]];

			be_processed[edge_to_adjacent[edge_refresh]] = false;
			edge_refresh = element_now0[another_points_index_now[0]] < edge_now[1] ? 
			element_now0[another_points_index_now[0]] * num_points + edge_now[1] : 
			edge_now[1] * num_points + element_now0[another_points_index_now[0]];

			be_processed[edge_to_adjacent[edge_refresh]] = false;
			edge_refresh = element_now1[another_points_index_now[1]] < edge_now[0] ? 
			element_now1[another_points_index_now[1]] * num_points + edge_now[0] : 
			edge_now[0] * num_points + element_now1[another_points_index_now[1]];

			be_processed[edge_to_adjacent[edge_refresh]] = false;
			edge_refresh = element_now1[another_points_index_now[1]] < edge_now[1] ? 
			element_now1[another_points_index_now[1]] * num_points + edge_now[1] : 
			edge_now[1] * num_points + element_now1[another_points_index_now[1]];

			be_processed[edge_to_adjacent[edge_refresh]] = false;

			be_processed[index] = true;
			index = 0;
			iter++;

			if (iter > num_edges * 10) {
				throw 1;
			}
		}
		else
		{
			be_processed[index] = true;
			index++;
		}
	}

	// return result
	for (int i = 0; i < num_elements; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			result[i * 3 + j] = elements[i * 3 + j];
		}
	}

	delete[] be_processed;
	return;
}