#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// problem size (vector length) N
static const int N = 12345678;

// Number of terms to use when approximating sine
static const int TERMS = 6;

// kernel function (CPU - Do not modify)
void sine_serial(float *input, float *output)
{
  int i;

  for (i=0; i<N; i++) {
      float value = input[i]; 
      float numer = input[i] * input[i] * input[i]; 
      int denom = 6; // 3! 
      int sign = -1; 
      for (int j=1; j<=TERMS;j++) 
      { 
         value += sign * numer / denom; 
         numer *= input[i] * input[i]; 
         denom *= (2*j+2) * (2*j+3); 
         sign *= -1; 
      } 
      output[i] = value; 
    }
}


// kernel function (CUDA device)
// TODO: Implement your graphics kernel here. See assignment instructions for method information

__global__ void sine_parallel(float *input, float *output)
{
	// dimension structure with fields
  int idx = blockIdx.x * 512 + threadIdx.x;
   
  float value = input[idx];
	// multiplying by 3
  float numer = input[idx] * input[idx] * input[idx];
  int denom = 6;
  int sign = -1;
  for (int i = 1; i<=TERMS; i++)
  {
	value += sign * numer / denom;
	numer *= input[idx] * input[idx];
	denom *= (2 * i + 2) * (2 * i + 3);
	sign *= -1;
  }
	// result writing into output array
  output[idx] = value;
}

// BEGIN: timing and error checking routines (do not modify)

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, std::string name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << std::setprecision(5);	
	std::cout << name << ": " << ((float) (end_time - start_time)) / (1000 * 1000) << " sec\n";
	return end_time - start_time;
}

void checkErrors(const char label[])
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

// END: timing and error checking routines (do not modify)



int main (int argc, char **argv)
{
  //BEGIN: CPU implementation (do not modify)
  float *h_cpu_result = (float*)malloc(N*sizeof(float));
  float *h_input = (float*)malloc(N*sizeof(float));
  //Initialize data on CPU
  int i;
  for (i=0; i<N; i++)
  {
    h_input[i] = 0.1f * i;
  }

  //Execute and time the CPU version
  long long CPU_start_time = start_timer();
  sine_serial(h_input, h_cpu_result);
  long long CPU_time = stop_timer(CPU_start_time, "\nCPU Run Time");
  //END: CPU implementation (do not modify)


  //TODO: Prepare and run your kernel, make sure to copy your results back into h_gpu_result and display your timing results
 
// instantiating constat values
  const int sizeOfBlock = 512;
  const int sizeOfGrid = N/512 + 1; 
  const float bytes = N * sizeof(float);
	
//Declaring the arrays  
  float *d_input;
  float *d_output;

	
  // timer for GPU start time
  long long GPU_startTotal = start_timer();
   
  //Allocating memory
  float *h_gpu_result = (float*)malloc(bytes);
  
  //Allocating memory to the GPU and timing it
  long long GPU_allocateStart = start_timer();
  cudaMalloc((void**) &d_input, bytes);
  cudaMalloc((void**) &d_output, bytes);
  long long GPU_allocateTime = stop_timer(GPU_allocateStart, "\nGPU Memory Allocation");

  // Copying input to output and timing it
  long long GPU_dcopyStart = start_timer();
  cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
  long long GPU_dcopyTime = stop_timer(GPU_dcopyStart, "Copying GPU Memory to Device");

  //Launching the kernel with N threads and then timing it
  long long GPU_kernelStart = start_timer();
  sine_parallel<<<sizeOfGrid, sizeOfBlock>>>(d_input, d_output);
  long long GPU_kernelTime = stop_timer(GPU_kernelStart, "GPU Kernel Run Time");

  //Copying the output of parallel_sine back to the host (h_gpu_result) and timing it
  long long GPU_hcopyStart = start_timer();
  cudaMemcpy(h_gpu_result, d_output, bytes, cudaMemcpyDeviceToHost);
  long long GPU_hcopyTime = stop_timer(GPU_hcopyStart, "Copying GPU Memory to Host");
	
  
  //Free GPU memory
  cudaFree(d_input);
  cudaFree(d_output);

  // End timer for GPU
  long long GPU_totalTime = stop_timer(GPU_startTotal, "Total GPU Run Time");

  // Checking to make sure the CPU and GPU results match - Do not modify
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_cpu_result[i]-h_gpu_result[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");

  // Cleaning up memory
  free(h_input);
  free(h_cpu_result);
  free(h_gpu_result);
  return 0;
}
