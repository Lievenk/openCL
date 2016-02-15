#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;
#define SHOW

// First MxN are input dimensions, M1xN1 are filter dimensions
// Transforms input matrix into matrix that makes the convolution easier
// squares are rows in transformed
float* transform(float* input, int M, int N, int M1, int N1) {
	float *transformed = (float *) malloc(sizeof(float)*M*N*M1*N1);
	int counter = 0;
	for(int i = 0; i<M; i++) {
		for(int j = 0; j<N; j++) {
			for(int k = i; k<i+M1; k++) {
				for(int l = j; l<j+N1; l++) {
					transformed[(i*N+j)*M1*N1+counter] = input[k*(N+2)+l];
					counter++;
				}
			}
			counter = 0;
		}
	}
	return transformed;
}

void print_clbuild_errors(cl_program program,cl_device_id device){
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

int main(int, char**)
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

	cl_mem input_a_buf;
	cl_mem input_b_buf; // num_devices elements
	cl_mem input_c_buf;
	cl_mem output_buf_1;
	cl_mem output_buf_2;
	int status;
//-------------------------------------------------

    
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

     unsigned char **opencl_program=read_file("dot_product.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "dot_prod", NULL); 

	// Input frame buffer
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       9*640*360*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

	// filter 1 of scharr edge detection
    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       9*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");
	
	// filter 2 of scharr edge detection
    input_c_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       9*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");
    
	// Filtered output frame buffer 1
    output_buf_1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 360*640*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

	// Filtered output frame buffer 2
    output_buf_2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 360*640*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[3];
	cl_event kernel_event,finish_event;

	
	float *input_b=(float *) malloc(sizeof(float)*9);
	float *input_c=(float *) malloc(sizeof(float)*9);

	// Sobel filters
	input_b[0] = 3;
	input_b[1] = 0;
	input_b[2] = -3;
	input_b[3] = 10;
	input_b[4] = 0;
	input_b[5] = -10;
	input_b[6] = 3;
	input_b[7] = 0;
	input_b[8] = -3;
	
	input_c[0] = 3;
	input_c[1] = 10;
	input_c[2] = 3;
	input_c[3] = 0;
	input_c[4] = 0;
	input_c[5] = 0;
	input_c[6] = -3;
	input_c[7] = -10;
	input_c[8] = -3;
	
	status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
        0, 9*sizeof(float), input_b, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input B");

	status = clEnqueueWriteBuffer(queue, input_c_buf, CL_FALSE,
        0, 9*sizeof(float), input_c, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input C");
    
	// Set kernel arguments.
    unsigned argi = 0;
	int N = 9;
	
	// Set the kernel arguments
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");
    
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_c_buf);
    checkError(status, "Failed to set argument 3");
    
	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf_1);
    checkError(status, "Failed to set argument 4");
	
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf_2);
    checkError(status, "Failed to set argument 5");
	
    status = clSetKernelArg(kernel, argi++, sizeof(int), &N);
    checkError(status, "Failed to set argument 6");

//-------------------------------------------------------------

	float *output1=(float *) malloc(640*360*sizeof(float));	
	float *output2=(float *) malloc(640*360*sizeof(float));	
    const size_t global_work_size = 640*360;

	VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;

    VideoWriter outputVideo;   // Open the output
    outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start,end;
	double diff,tot = 0;
	int count=0;
	Mat dst;
	Mat dst2;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif
    while (true) {
        Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 200) break; // 299
        camera >> cameraFrame;
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);	
        Mat grayframe,edge_x,edge_y,edge,edge_inv;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
		
		Mat edge_x_ref, edge_y_ref, edge_ref;
		
		// Blur the image
		GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		//--------------------
		// Pad the frame
		copyMakeBorder(grayframe,dst,1,1,1,1,BORDER_CONSTANT,0);
		// Convert to float
		dst.convertTo(dst2,CV_32FC1);
		// Transform the matrix for easy convolution
		float *trans = transform((float*)dst2.data,360,640,3,3);
		time (&start);
		// Write the transformed frame to the gpu buffer
		status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
			0, 9*640*360*sizeof(float), trans, 0, NULL, &write_event[2]);
		checkError(status, "Failed to transfer input A");
		// Apply sobel edge detection filter
		status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        	&global_work_size, NULL, 1, write_event, &kernel_event);
    	checkError(status, "Failed to launch kernel");
		// Read the first filtered frame
    	status = clEnqueueReadBuffer(queue, output_buf_1, CL_TRUE,
        	0, 640*360*sizeof(float), output1, 1, &kernel_event, &finish_event);
    	// Read the second filtered frame
		status = clEnqueueReadBuffer(queue, output_buf_2, CL_TRUE,
        	0, 640*360*sizeof(float), output2, 1, &kernel_event, &finish_event);
		//----------------------
		
		// Put the outputs in Mat objects
		edge_x = cv::Mat(360,640,CV_32FC1,output1);
		edge_y = cv::Mat(360,640,CV_32FC1,output2);

		// Convert to unsigned char
		edge_x.convertTo(edge_x,CV_8UC1);
		edge_y.convertTo(edge_y,CV_8UC1);

		addWeighted(edge_x,0.5,edge_y,0.5,0,edge);
        threshold(edge,edge,80,255,THRESH_BINARY_INV);

		time (&end);
        cvtColor(edge, edge_inv, CV_GRAY2BGR);
    	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		grayframe.copyTo(displayframe,edge);
        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
		diff = difftime (end,start);
		tot+=diff;
	}
	outputVideo.release();
	camera.release();
  	printf ("FPS %.2lf .\n", 299.0/tot );
		
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(write_event[2]);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_a_buf);	
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(input_c_buf);
	clReleaseMemObject(output_buf_1);
	clReleaseMemObject(output_buf_2);
	clReleaseProgram(program);
	clReleaseContext(context);
	
    clFinish(queue);
    
	return EXIT_SUCCESS;

}
