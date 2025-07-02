#include "pch.h"
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <omp.h>
using namespace std;

typedef boost::multiprecision::cpp_bin_float_50 lf;
double t[6]{ 0 };
struct triangle
{
	int ind[3]{-1,-1,-1};
	double *nodes;
	lf radius;
	double xmax, xmin, ymax, ymin;
	lf x, y;
	triangle(int a, int b, int c, double* nodes) {
		this->ind[0] = a;
		this->ind[1] = b;
		this->ind[2] = c;
		this->nodes = nodes;
		
		lf x0 = nodes[a * 2];
		lf y0 = nodes[a * 2 + 1];
		lf x1 = nodes[b * 2];
		lf y1 = nodes[b * 2 + 1];
		lf x2 = nodes[c * 2];
		lf y2 = nodes[c * 2 + 1];

		// calculate the circumcenter
		lf d = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1));
		this->x = ((x0 * x0 + y0 * y0) * (y1 - y2) + (x1 * x1 + y1 * y1) * (y2 - y0) + (x2 * x2 + y2 * y2) * (y0 - y1)) / d;
		this->y = ((x0 * x0 + y0 * y0) * (x2 - x1) + (x1 * x1 + y1 * y1) * (x0 - x2) + (x2 * x2 + y2 * y2) * (x1 - x0)) / d;


		// calculate the radius
		lf s0 = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
		lf s1 = sqrt((x0-x2)*(x0-x2) + (y0-y2)*(y0-y2));
		lf s2 = sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1));
		lf s = (s0 + s1 + s2) / 2;
		this->radius = s0 * s1 * s2 / (4 * sqrt(s * (s - s0) * (s - s1) * (s - s2)));

		this->xmax = (this->x + radius).convert_to <double>();
		this->xmin = (this->x - radius).convert_to <double>();
		this->ymax = (this->y + radius).convert_to <double>();
		this->ymin = (this->y - radius).convert_to <double>();
	}

	bool isin(int ind_node){

		double t0 = clock();
		// bounding box check
		if (nodes[ind_node * 2] < this->xmin || nodes[ind_node * 2] > this->xmax || nodes[ind_node * 2 + 1] < this->ymin || nodes[ind_node * 2 + 1] > this->ymax) {
			return false;
		}

		// distance check
		lf s = sqrt((nodes[ind_node * 2] - x) * (nodes[ind_node * 2] - x) + (nodes[ind_node * 2 + 1] - y) * (nodes[ind_node * 2 + 1] - y));
		double t1 = clock();
		bool result = s < radius;
		double t2 = clock();

		t[4] += (t1 - t0) / CLOCKS_PER_SEC;
		t[5] += (t2 - t1) / CLOCKS_PER_SEC;
		return result;
	}
};



vector<triangle> mesh;

void Bowyer_Watson(int num_nodes, double* nodes, int ind_node) {
	// find the triangles that contain the new node
	double t0 = clock();
	vector<int> ind_tri;

	omp_set_num_threads(8);
//#pragma omp parallel for
	for (int i = 0; i < mesh.size(); i++) {
		bool cond = mesh[i].isin(ind_node);

		if (cond) {
			ind_tri.push_back(i);
		}
	}

	// find the edges that are shared by the triangles
	double t1 = clock();
	map<pair<int, int>, int> num_edge;
	for (int i = 0; i < ind_tri.size(); i++) {
		for (int j = 0; j < 3; j++) {
			int a = mesh[ind_tri[i]].ind[j];
			int b = mesh[ind_tri[i]].ind[(j + 1) % 3];
			if (a > b) {
				swap(a, b);
			}
			num_edge[make_pair(a, b)]++;
		}
	}

	// remove the triangles that contain the new node
	double t2 = clock();
	sort(ind_tri.begin(), ind_tri.end(), greater<int>());
	for (int i = 0; i < ind_tri.size(); i++) {
		mesh.erase(mesh.begin() + ind_tri[i]);
	}

	// add the new triangles
	double t3 = clock();
	for (auto it = num_edge.begin(); it != num_edge.end(); it++) {
		if (it->second == 1) {
			mesh.push_back(triangle(it->first.first, it->first.second, ind_node, nodes));
		}
	}
	double t4 = clock();

	t[0] += (t1 - t0) / CLOCKS_PER_SEC;
	t[1] += (t2 - t1) / CLOCKS_PER_SEC;
	t[2] += (t3 - t2) / CLOCKS_PER_SEC;
	t[3] += (t4 - t3) / CLOCKS_PER_SEC;
}

int argmax(lf* arr, int num_arr) {
	int result = 0;
	for (int i = 0; i < num_arr; i++) {
		if (arr[i] > arr[result]) {
			result = i;
		}
	}
	return result;
}

void triangular_mesh(int* num_mesh, double* nodes, int num_nodes) {
	
	// clear the mesh
	mesh.clear();

	// get the bouding box
	double maxX = -1e10;
	double maxY = -1e10;
	double minX = 1e10;
	double minY = 1e10;
	for (int i = 0; i < num_nodes; i++) {
		maxX = max(maxX, nodes[i * 2]);
		maxY = max(maxY, nodes[i * 2 + 1]);
		minX = min(minX, nodes[i * 2]);
		minY = min(minY, nodes[i * 2 + 1]);
	}

	// initialize the bounding box
	double* nodes2 = new double[(num_nodes + 4) * 2];
	memcpy(nodes2, nodes, sizeof(double) * 2 * num_nodes);
	maxX += 1;
	maxY += 1;
	minX -= 1;
	minY -= 1;
	nodes2[num_nodes * 2] = maxX;
	nodes2[num_nodes * 2 + 1] = maxY;

	nodes2[(num_nodes + 1) * 2] = minX;
	nodes2[(num_nodes + 1) * 2 + 1] = maxY;

	nodes2[(num_nodes + 2) * 2] = minX;
	nodes2[(num_nodes + 2) * 2 + 1] = minY;

	nodes2[(num_nodes + 3) * 2] = maxX;
	nodes2[(num_nodes + 3) * 2 + 1] = minY;

	mesh.push_back(triangle(num_nodes, num_nodes+1, num_nodes+2, nodes2));
	mesh.push_back(triangle(num_nodes, num_nodes+2, num_nodes+3, nodes2));

	num_nodes += 4;
	
	// add the nodes one by one
	for (int i = 0; i < num_nodes-4; i++) {
		Bowyer_Watson(num_nodes, nodes2, i);
	}

	num_nodes -= 4;

	// delete the bounding box
	for (int i=0;i<mesh.size();i++){
		for (int j=0;j<3;j++){
			if (mesh[i].ind[j] >= num_nodes){
				mesh.erase(mesh.begin() + i);
				i--;
				break;
			}
		}
	}

	// output the result
	num_mesh[0] = mesh.size();
	delete[] nodes2;

	//printf("time: %lf %lf %lf %lf %lf %lf\n", t[0], t[1], t[2], t[3], t[4], t[5]);

}

void get_triangular_mesh(int* results){
	for (int i=0;i<mesh.size();i++){
		results[i*3] = mesh[i].ind[0];
		results[i*3+1] = mesh[i].ind[1];
		results[i*3+2] = mesh[i].ind[2];
	}
}