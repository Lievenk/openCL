#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <chrono>
#define STRING_BUFFER_LEN 1024
using namespace std;


void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}
unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  return output;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------
const unsigned N = 256;
float *input_a=(float *) malloc(sizeof(float)*N);
//float *output=(float *) malloc(sizeof(float)*N/);
float *ref_output=(float *) malloc(sizeof(float));
cl_mem input_a_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;

	//time_t start,end;
	//clock_t start,end;
	double diff;
	//auto start = std::chrono::high_resolution_clock::now();
	
	for(unsigned j = 0; j < N; ++j) {	
	      input_a[j] = j;//rand_float();
	}

	auto start = std::chrono::high_resolution_clock::now();
	for(unsigned j = 0; j < N; ++j) {
	      *ref_output = input_a[j] + *ref_output;
	      //printf("ref %f\n",ref_output[j]);
	    }
	*ref_output = *ref_output/((float)N);
	auto end = std::chrono::high_resolution_clock::now();
	diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0;
	printf ("CPU result %f\n", *ref_output);
  	printf ("CPU took %.3lf seconds to run.\n", diff );

//	start = std::chrono::high_resolution_clock::now();
     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("vector_avg.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_avg", NULL);
	
	size_t max_workgroup_size;
	clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&max_workgroup_size,NULL);
	printf("Workgroup size:%i \n",max_workgroup_size);
 
	// Input buffer.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

	//int output_size = ceil((float)N/(float)max_workgroup_size)*sizeof(float);
    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

	float *output=(float *) malloc(4*sizeof(float));


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
	cl_event kernel_event,finish_event;
    
	status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        0, N* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 2");
	
	int group_size = 64;
    status = clSetKernelArg(kernel, argi++, sizeof(int), &group_size);
    checkError(status, "Failed to set argument 3");

    const size_t global_work_size = 4;
 
	start = std::chrono::high_resolution_clock::now();
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_work_size, NULL, 1, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
	end = std::chrono::high_resolution_clock::now();
    
	// Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, 4* sizeof(float), output, 1, &kernel_event, &finish_event);

	diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
   printf ("GPU took %.3lf seconds to run.\n", diff/1000.0);
// Verify results.
float outputAvg = 0;
for(int i = 0; i<4; i++) {
	outputAvg += output[i];
	//printf("Output %d: %f\n",i,output[i]);
}
outputAvg = outputAvg/256;

if(fabsf(outputAvg - *ref_output) > 1.0e-5f) {
   printf("Failed verification\nOutput: %f\nReference: %f\n", outputAvg, *ref_output);
} else {
	printf("GPU average is correct\n");
}
    // Release local events.
    clReleaseEvent(write_event[0]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);


//--------------------------------------------------------------------

     clFinish(queue);

     return 0;
}
