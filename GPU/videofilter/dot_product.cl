__kernel void dot_prod(__global const float *frame,__global const float *filter1,__global const float *filter2,
	__global float *restrict output1,__global float *restrict output2, int N)
{
	int threadID = get_global_id(0);
	output1[threadID] = 0;
	output2[threadID] = 0;

	for(int i = 0; i < N; i++) {
		output1[threadID] += frame[N*threadID+i] * filter1[i];
		output2[threadID] += frame[N*threadID+i] * filter2[i];	
	}
}

