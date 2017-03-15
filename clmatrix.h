#ifndef CLMATRIX_H
#define CLMATRIX_H

#include <eigen3/Eigen/Dense>
#include "clutil.h"

class CLMatrix {
    private:
        float* data;
        cl::Buffer buffer;
        long long rows;
        long long cols;
    public:
        CLMatrix(cl::Context context, cl::CommandQueue queue, long long rows, long long cols);
        float* getData();
        cl::Buffer getBuffer();
        Eigen::MatrixXf toMatrixXf();
};

#endif // CLMATRIX_H
