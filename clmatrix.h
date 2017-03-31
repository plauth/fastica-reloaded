#ifndef CLMATRIX_H
#define CLMATRIX_H

#include <eigen3/Eigen/Dense>
#include <cstring>
#include "clutil.h"

class CLMatrix {
    private:
        float* data;
        cl::Buffer buffer;
        cl::CommandQueue queue;
        long long rows;
        long long cols;
    public:
        CLMatrix(cl::Context context, cl::CommandQueue queue, long long rows, long long cols);
        ~CLMatrix();
        float* getData();
        cl::Buffer getBuffer();
        Eigen::MatrixXf toMatrixXf();
        Eigen::VectorXf toVectorXf();
        long long getRows();
        long long getCols();
        void setZero();
        void setValue(float value);
};

#endif // CLMATRIX_H
