#include "../include/triangle.h"
#include <cmath>

Triangle::Triangle(int point_a, int point_b, int point_c, std::span<const double> nodes_span) 
    : nodes(nodes_span) {
    ind[0] = point_a;
    ind[1] = point_b;
    ind[2] = point_c;
    
    lf x0 = nodes[point_a * 2];
    lf y0 = nodes[point_a * 2 + 1];
    lf x1 = nodes[point_b * 2];
    lf y1 = nodes[point_b * 2 + 1];
    lf x2 = nodes[point_c * 2];
    lf y2 = nodes[point_c * 2 + 1];

    // calculate the circumcenter
    lf d = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1));
    x = ((x0 * x0 + y0 * y0) * (y1 - y2) + (x1 * x1 + y1 * y1) * (y2 - y0) + 
         (x2 * x2 + y2 * y2) * (y0 - y1)) / d;
    y = ((x0 * x0 + y0 * y0) * (x2 - x1) + (x1 * x1 + y1 * y1) * (x0 - x2) + 
         (x2 * x2 + y2 * y2) * (x1 - x0)) / d;

    // calculate the radius
    lf s0 = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    lf s1 = std::sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2));
    lf s2 = std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    lf s = (s0 + s1 + s2) / 2;
    radius = s0 * s1 * s2 / (4 * std::sqrt(s * (s - s0) * (s - s1) * (s - s2)));

    xmax = x + radius;
    xmin = x - radius;
    ymax = y + radius;
    ymin = y - radius;
}

bool Triangle::isInCircumcircle(int ind_node) const {
    // bounding box check
    if (nodes[ind_node * 2] < xmin || nodes[ind_node * 2] > xmax || 
        nodes[ind_node * 2 + 1] < ymin || nodes[ind_node * 2 + 1] > ymax) {
        return false;
    }

    // distance check
    lf s = std::sqrt((nodes[ind_node * 2] - x) * (nodes[ind_node * 2] - x) + 
                     (nodes[ind_node * 2 + 1] - y) * (nodes[ind_node * 2 + 1] - y));
    return s < radius;
}
