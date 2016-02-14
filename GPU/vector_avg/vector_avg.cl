__kernel void vector_avg(__global const float *x,__global float *restrict z, int group_size)
{
	int threadID = get_global_id(0);
	float total = 0;

	for(int i = 0; i < group_size; i++) {
		total += x[threadID*group_size + i];
	}
	z[threadID] = total;
}




