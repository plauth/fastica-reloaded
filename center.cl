__kernel void center(__global float *minuends, __global float *subtrahends, unsigned int signal_length) {
    unsigned int index = (get_group_id(1)*signal_length) + (get_group_id(0) * get_local_size(0) * 8 + get_local_id(0));

    float subtrahend =  subtrahends[get_group_id(1)];
    int offset = 0;

    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);
    minuends[index + offset] -= subtrahend; offset += get_local_size(0);

}
