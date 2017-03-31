#include "clcenterkernel.h"

using namespace std;

std::shared_ptr<CLMatrix> CLCenterKernel::subtract(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> minuend, std::shared_ptr<CLMatrix> subtrahend) {
    static std::once_flag    compileFlag;

    static cl::Program centerProg;
    static cl::Kernel centerKernel;

    std::call_once(compileFlag, [&context]() {
        // load opencl source
        ifstream cl_file("center.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));

        centerProg = cl::Program(context, cl_string.c_str());
        cl_int err = centerProg.build();
        checkErr(err, "cl::Programm::build()");

        centerKernel = cl::Kernel(centerProg, "center", &err);
        checkErr(err, "cl::Kernel()");
    });

    cl_int err;
    err = centerKernel.setArg(0, minuend->getBuffer());
    checkErr(err, "Kernel::setArg(0)");
    err = centerKernel.setArg(1, subtrahend->getBuffer());
    checkErr(err, "Kernel::setArg(1)");
    unsigned int signal_length = minuend->getCols();
    err = centerKernel.setArg(2, sizeof(unsigned int), &signal_length);
    checkErr(err, "Kernel::setArg(2)");

    size_t workgroup_size = getMaxWorkGroupSize(context);

    err = queue.enqueueNDRangeKernel(centerKernel, cl::NullRange, cl::NDRange(minuend->getCols()/8,minuend->getRows()), cl::NDRange(workgroup_size, 1));
    checkErr(err, "enqueueNDRangeKernel()");

    return minuend;
}

