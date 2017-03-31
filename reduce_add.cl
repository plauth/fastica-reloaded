#define N_IS_POW_2 1

__kernel void reduce_add(__global float *input, __global float *output, __local float *temp, unsigned int signal_length) {
    unsigned int tid = get_local_id(0);
    unsigned int i = (get_group_id(1)*signal_length) + (get_group_id(0)*(WORKGROUP_SIZE*2) + get_local_id(0));
    unsigned int numThreads = WORKGROUP_SIZE*2*get_num_groups(0);
    temp[tid] = 0;

    while (i < (get_group_id(1)+1)*signal_length)
    {
        temp[tid] += input[i];
        if (N_IS_POW_2 || i + WORKGROUP_SIZE < (get_group_id(1)+1)*signal_length)
            temp[tid] += input[i+WORKGROUP_SIZE];
        i += numThreads;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    if (WORKGROUP_SIZE >= 8192) { if (tid < 4096) { temp[tid] += temp[tid + 4096]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >= 4096) { if (tid < 2048) { temp[tid] += temp[tid + 2048]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >= 2048) { if (tid < 1024) { temp[tid] += temp[tid + 1024]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >= 1024) { if (tid <  512) { temp[tid] += temp[tid +  512]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=  512) { if (tid <  256) { temp[tid] += temp[tid +  256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=  256) { if (tid <  128) { temp[tid] += temp[tid +  128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=  128) { if (tid <   64) { temp[tid] += temp[tid +   64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=   64) { if (tid <   32) { temp[tid] += temp[tid +   32]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=   32) { if (tid <   16) { temp[tid] += temp[tid +   16]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=   16) { if (tid <    8) { temp[tid] += temp[tid +    8]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=    8) { if (tid <    4) { temp[tid] += temp[tid +    4]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=    4) { if (tid <    2) { temp[tid] += temp[tid +    2]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (WORKGROUP_SIZE >=    2) { if (tid <    1) { temp[tid] += temp[tid +    1]; } barrier(CLK_LOCAL_MEM_FENCE); }

    // write result for this block to global mem
    if (tid == 0) output[get_group_id(1) * get_local_size(0) + get_group_id(0)] = temp[0];
    //if (tid == 0) output[get_group_id(1) * get_local_size(0) + get_group_id(0)] = i;
}
