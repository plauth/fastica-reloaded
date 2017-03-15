#include "clreduceaddkernel.h"

static template<typename T> CLMatrix CLReduceAddKernel::exec(cl::Context context, cl::CommandQueue queue, CLMatrix matrix)
{
    static std::once_flag    compileFlag;

    static cl::Program        addProg;
    static cl::Kernel        addkernel;

    std::call_once(compileFlag, []() {
        std::string kern = "__kernel void add(global const T * const a, global const T * const b, global T * restrict const c) { unsigned idx = get_global_id(0); c[idx] = a[idx] + b[idx]; }";

        std::ostringstream options;
        options<<"-D BLOCK_SIZE="<< CLTypes<T>::getName();

        addProg        = cl::Program(kern, false);
        addProg.build(options.str().c_str());
        std::cout<<"vector addition kernel compiled for type: "<<CLTypes<T>::getName()<<std::endl;
        addkernel    = cl::Kernel(addProg, "add");
    });

    Buffer a(begin(in1), end(in1), true, false);
    Buffer b(begin(in2), end(in2), true, false);
    Buffer c(CL_MEM_READ_WRITE, nElems * sizeof(T));

    auto addOp = cl::make_kernel<Buffer, Buffer, Buffer>(addkernel);

    addOp(EnqueueArgs(nElems), a, b, c);

    cl::copy(c, begin(out), end(out));
}
