#ifndef CLCENTERKERNEL_H
#define CLCENTERKERNEL_H

#include <fstream>
#include <iostream>
#include "clmatrix.h"
#include "clutil.h"

class CLCenterKernel
{
    private:
        CLCenterKernel();
    public:
        static std::shared_ptr<CLMatrix> subtract(cl::Context context, cl::CommandQueue queue, std::shared_ptr<CLMatrix> minuend, std::shared_ptr<CLMatrix> subtrahend);

};

#endif // CLCENTERKERNEL_H
