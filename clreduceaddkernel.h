#ifndef CLREDUCEADDKERNEL_H
#define CLREDUCEADDKERNEL_H

#include <fstream>
#include <iostream>
#include "cl2.hpp"
#include "clmatrix.h"
#include "clutil.h"

using namespace std;

class CLReduceAddKernel
{
    private:
        CLReduceAddKernel();
    public:
        template<unsigned int BLOCK_SIZE> static CLMatrix exec(cl::Context context, cl::CommandQueue queue, CLMatrix matrix);
};

template<unsigned int BLOCK_SIZE> CLMatrix CLReduceAddKernel::exec(cl::Context context, cl::CommandQueue queue, CLMatrix matrix) {
    static std::once_flag    compileFlag;

    static cl::Program reduceAddProg;
    static cl::Kernel reducedAddkernel;

    std::call_once(compileFlag, [&context]() {
        // load opencl source
        ifstream cl_file("reduce_add.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        //cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

        std::ostringstream options;
        options<<"-D BLOCK_SIZE="<< BLOCK_SIZE;

        reduceAddProg = cl::Program(context, cl_string.c_str());
        //add devices here?
        cl_int err = reduceAddProg.build(options.str().c_str());
        checkErr(err, "cl::Programm::build()");

        std::cout<<"reduce_add kernel compiled for block size: "<< BLOCK_SIZE <<std::endl;
        reducedAddkernel = cl::Kernel(reduceAddProg, "add");
    });

    //Buffer a(begin(in1), end(in1), true, false);
    //Buffer b(begin(in2), end(in2), true, false);
    //Buffer c(CL_MEM_READ_WRITE, nElems * sizeof(T));

    //auto addOp = cl::make_kernel<Buffer, Buffer, Buffer>(addkernel);

    //addOp(EnqueueArgs(nElems), a, b, c);

    //cl::copy(c, begin(out), end(out));

    return matrix;
}

#endif // CLREDUCEADDKERNEL_H
