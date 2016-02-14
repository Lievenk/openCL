__kernel void matrix_mult(__global const float *x,__global const float *y,__global float *restrict z, int N, int O)
{
	int threadID = get_global_id(0);
	float total = 0;
	int row = threadID/O;
	int column = threadID%O;
	
	for(int i = 0; i < N; i++) {
		total += x[row*N + i] * y[i*O + column];
	}
	z[threadID] = total;
}







