__kernel void dot_prod(__global const float *x,__global const float *y,__global float *restrict z, int N)
{
	int threadID = get_global_id(0);
	z[threadID] = 0;
	for(int i = 0; i < N; i++) {
		z[threadID] += x[N*threadID+i] * y[i];
	}
}







