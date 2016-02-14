__kernel void vector_avg(__global const float *x,__global float restrict z)
{
	int groupID = get_group_id(0);
	int localID = get_local_id(0);
	int localSize = get_local_size*(0);

	z = x[groupID*localSize + localID];
}
