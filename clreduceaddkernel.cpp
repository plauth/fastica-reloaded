#include "clreduceaddkernel.h"

using namespace Eigen;

inline int retrieveWorkgroupCount(std::shared_ptr<CLMatrix> input, int workitems_per_workgroup) {
    int work_per_workitem = 1;
    while(input->getCols() / workitems_per_workgroup / work_per_workitem > workitems_per_workgroup) {
        work_per_workitem <<= 1;
    }

    return input->getCols() / workitems_per_workgroup / work_per_workitem;
}

std::shared_ptr<CLMatrix> CLReduceAddKernel::getMeans(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> input) {
    std::shared_ptr<CLMatrix> result = std::make_shared<CLMatrix>(context, queue, input->getRows(), 1);
    std::shared_ptr<CLMatrix> kernelOutput = CLReduceAddKernel::exec(context, queue, input);
    checkErr(queue.finish(), "clFinish()");

    Eigen::MatrixXf resultMatrix = kernelOutput->toMatrixXf().rowwise().sum() / input->getCols();
    for(int row = 0; row < input->getRows(); row++) {
        result->getData()[row] = resultMatrix(row,0);
    }

    return result;
}

std::shared_ptr<CLMatrix> CLReduceAddKernel::exec(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> input) {
    std::shared_ptr<CLMatrix> output;
    int num_workgroups;

    switch(getMaxWorkGroupSize(context)) {
    case 8192:
        num_workgroups = retrieveWorkgroupCount(input, 8192);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<8192>(context, queue, input, output, num_workgroups);
        break;
    case 4096:
        num_workgroups = retrieveWorkgroupCount(input, 4096);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<4096>(context, queue, input, output, num_workgroups);
        break;
    case 2048:
        num_workgroups = retrieveWorkgroupCount(input, 2048);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<2048>(context, queue, input, output, num_workgroups);
        break;
    case 1024:
        num_workgroups = retrieveWorkgroupCount(input, 1024);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<1024>(context, queue, input, output, num_workgroups);
        break;
    case 512:
        num_workgroups = retrieveWorkgroupCount(input, 512);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<512>(context, queue, input, output, num_workgroups);
        break;
    case 256:
        num_workgroups = retrieveWorkgroupCount(input, 256);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<256>(context, queue, input, output, num_workgroups);
        break;
    case 128:
        num_workgroups = retrieveWorkgroupCount(input, 128);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<128>(context, queue, input, output, num_workgroups);
        break;
    case 64:
        num_workgroups = retrieveWorkgroupCount(input, 64);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<64>(context, queue, input, output, num_workgroups);
        break;
    case 32:
        num_workgroups = retrieveWorkgroupCount(input, 32);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<32>(context, queue, input, output, num_workgroups);
        break;
    case 16:
        num_workgroups = retrieveWorkgroupCount(input, 16);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<16>(context, queue, input, output, num_workgroups);
        break;
    case 8:
        num_workgroups = retrieveWorkgroupCount(input, 8);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<8>(context, queue, input, output, num_workgroups);
        break;
    case 4:
        num_workgroups = retrieveWorkgroupCount(input, 4);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<4>(context, queue, input, output, num_workgroups);
        break;
    case 2:
        num_workgroups = retrieveWorkgroupCount(input, 2);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<2>(context, queue, input, output, num_workgroups);
        break;
    case 1:
        num_workgroups = retrieveWorkgroupCount(input, 1);
        output = std::make_shared<CLMatrix>(context, queue, input->getRows(), num_workgroups);
        CLReduceAddKernel::launch_kernel<1>(context, queue, input, output, num_workgroups);
        break;
    default:
        std::cerr << "ERROR: Your compute device seems to support a maximum workgroup size that we are not prepared for :-/" << std::endl;
        exit(EXIT_FAILURE);
    }

    return output;
}
