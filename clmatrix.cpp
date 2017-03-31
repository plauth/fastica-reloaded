#include "clmatrix.h"


using namespace Eigen;

CLMatrix::CLMatrix(cl::Context context, cl::CommandQueue queue, long long rows, long long cols) {
    this->queue = queue;
    this->rows = rows;
    this->cols = cols;
    std::size_t size = rows * cols * sizeof(float);
    this->buffer = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size);
    cl_int err;
    this->data = (float*) this->queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, NULL, NULL, &err);
    checkErr(err, "clEnqueueMapBuffer()");
    checkErr(queue.finish(), "clEnqueueMapBuffer() --> clFinish()");
}

CLMatrix::~CLMatrix() {
    //std::cout << "Calling destructor of CLMatrix" << std::endl;
    cl_int err = this->queue.enqueueUnmapMemObject(this->buffer, this->data);
    checkErr(err, "enqueueUnmapMemObject()");
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

VectorXf CLMatrix::toVectorXf() {
    return Map<VectorXf> (this->data, this->rows);
}

long long CLMatrix::getRows() {
    return this->rows;
}

long long CLMatrix::getCols() {
    return this->cols;
}

void CLMatrix::setZero() {
    std::memset(this->data, 0,  this->rows * this->cols * sizeof(float));
}

void CLMatrix::setValue(float value) {
    for (int row = 0; row < this->getRows(); row++) {
        for (int col = 0; col < this->getCols(); col++) {
            this->data[row*this->cols + col] = value;
        }
    }
}
