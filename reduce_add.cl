//#ifndef BLOCK_SIZE
//#define BLOCK_SIZE 512
//#endif

__kernel void reduce_add(__global float *input, __global float *output, __local float *temp, unsigned int signal_length, unsigned int workspace_length) {
    unsigned int row = get_local_id(1);
    unsigned int tid = get_local_id(0) + row*BLOCK_SIZE;

    unsigned int numThreads = get_local_size(0) * get_num_groups(0);
    unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0) + row*signal_length;

    float localSum = 0.0;

    for ( ; i < signal_length*(row+1); i += numThreads ) {
        localSum += input[i];
    }

    temp[tid] = localSum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (BLOCK_SIZE >= 512) { if (get_local_id(0) < 256) { temp[tid] = localSum = localSum + temp[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 256) { if (get_local_id(0) < 128) { temp[tid] = localSum = localSum + temp[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 128) { if (get_local_id(0) <  64) { temp[tid] = localSum = localSum + temp[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 64 ) { if (get_local_id(0) <  32) { temp[tid] = localSum = localSum + temp[tid +  32]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 32 ) { if (get_local_id(0) <  16) { temp[tid] = localSum = localSum + temp[tid +  16]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 16 ) { if (get_local_id(0) <   8) { temp[tid] = localSum = localSum + temp[tid +   8]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 8  ) { if (get_local_id(0) <   4) { temp[tid] = localSum = localSum + temp[tid +   4]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 4  ) { if (get_local_id(0) <   2) { temp[tid] = localSum = localSum + temp[tid +   2]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (BLOCK_SIZE >= 2  ) { if (get_local_id(0) <   1) { temp[tid] = localSum = localSum + temp[tid +   1]; } barrier(CLK_LOCAL_MEM_FENCE); }

    if (get_local_id(0) == 0) { output[get_group_id(0) + get_local_id(1)*workspace_length] = temp[tid]; }

}
