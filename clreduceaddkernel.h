#ifndef CLREDUCEADDKERNEL_H
#define CLREDUCEADDKERNEL_H

#include <fstream>
#include <iostream>
//#include "cl2.hpp"
#include "clmatrix.h"
#include "clutil.h"

using namespace std;

class CLReduceAddKernel
{
    private:
        CLReduceAddKernel();
        template<unsigned int WORKGROUP_SIZE> static void launch_kernel(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> input, std::shared_ptr<CLMatrix> workspace, int num_workgroups);
        static std::shared_ptr<CLMatrix> exec(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> matrix);
    public:
        static std::shared_ptr<CLMatrix> getMeans(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> input);

};

template<unsigned int WORKGROUP_SIZE> void CLReduceAddKernel::launch_kernel(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> input, std::shared_ptr<CLMatrix> workspace, int num_workgroups) {
    static std::once_flag    compileFlag;

    static cl::Program reduceAddProg;
    static cl::Kernel reduceAddKernel;

    std::call_once(compileFlag, [&context]() {
        // load opencl source
        ifstream cl_file("reduce_add.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));

        std::ostringstream options;
        options<<"-D WORKGROUP_SIZE="<< WORKGROUP_SIZE;

        reduceAddProg = cl::Program(context, cl_string.c_str());
        //add devices here?
        cl_int err = reduceAddProg.build(options.str().c_str());
        checkErr(err, "cl::Programm::build()");

        reduceAddKernel = cl::Kernel(reduceAddProg, "reduce_add", &err);
        checkErr(err, "cl::Kernel()");
    });

    cl_int err;
    err = reduceAddKernel.setArg(0, input->getBuffer());
    checkErr(err, "Kernel::setArg(0)");
    err = reduceAddKernel.setArg(1, workspace->getBuffer());
    checkErr(err, "Kernel::setArg(1)");
    //local memory allocation
    err = reduceAddKernel.setArg(2, 2 * WORKGROUP_SIZE * sizeof(float), NULL);
    checkErr(err, "Kernel::setArg(2)");
    unsigned int signal_length = input->getCols();
    err = reduceAddKernel.setArg(3, sizeof(unsigned int), &signal_length);
    checkErr(err, "Kernel::setArg(3)");

    err = queue.enqueueNDRangeKernel(reduceAddKernel, cl::NullRange, cl::NDRange(input->getCols()/num_workgroups,input->getRows()), cl::NDRange(WORKGROUP_SIZE, 1));
    checkErr(err, "enqueueNDRangeKernel()");
}

#endif // CLREDUCEADDKERNEL_H
