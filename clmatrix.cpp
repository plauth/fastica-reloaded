#include "clmatrix.h"

using namespace Eigen;

CLMatrix::CLMatrix(cl::Context context, cl::CommandQueue queue, long long rows, long long cols) {
    this->rows = rows;
    this->cols = cols;
    std::size_t size = rows * cols * sizeof(float);
    this->buffer = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size);
    cl_int err;
    this->data = (float*) queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, NULL, NULL, &err);
    checkErr(err, "clEnqueueMapBuffer()");
    checkErr(queue.finish(), "clEnqueueMapBuffer() --> clFinish()");
}

float* CLMatrix::getData() {
    return this->data;
}

cl::Buffer CLMatrix::getBuffer() {
    return this->buffer;
}

MatrixXf CLMatrix::toMatrixXf() {
    return Map<Matrix<float,Dynamic,Dynamic,RowMajor>> (this->data, this->rows, this->cols);
}
