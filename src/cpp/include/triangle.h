#pragma once

#include <span>

using lf = double;

class Triangle {
private:
    lf radius;
    double xmax, xmin, ymax, ymin;
    lf x, y;
    std::span<const double> nodes;

public:
    int ind[3]{-1, -1, -1};

    Triangle(int point_a, int point_b, int point_c, std::span<const double> nodes_span);
    
    bool isInCircumcircle(int ind_node) const;
};
