/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

void testToneMapping();


__global__ void reduceLuminance(const float* const d_logLuminance, bool isMax, float *result, int size)
{
//	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	// load shared mem from global mem
	if(myId<size)
		sdata[tid] = d_logLuminance[myId];
	__syncthreads();            // make sure entire block is loaded!

	float identity=isMax?0:9999999999999;

	// do reduction in shared mem
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			float val1=sdata[tid];
			float val2=myId+s>size?identity:sdata[tid + s];
			if(isMax)
			{
				if(val1>val2)
					sdata[tid]=val1;
				else
					sdata[tid]=val2;
			}
			else
			{
				if(val1<val2)
					sdata[tid]=val1;
				else
					sdata[tid]=val2;
			}
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		result[blockIdx.x] = sdata[0];
	}
}

__global__ void assignHistogram(int *d_bins, const float *d_in, int BIN_COUNT, float lumRange, float lumMin)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    //bin = (lum[i] - lumMin) / lumRange * numBins
    int myBin = ((float)(d_in[myId] - lumMin) / (float)lumRange * BIN_COUNT);
    if(myBin>=BIN_COUNT)
    	myBin=BIN_COUNT-1;
    atomicAdd(&(d_bins[myBin]), 1);
}


__global__ void scanHistogram(const int* const d_bins, unsigned int *result, int size)
{
//	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ int sintdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	// load shared mem from global mem
	if(myId<size)
		sintdata[tid] = d_bins[myId];
	__syncthreads();            // make sure entire block is loaded!

	//Step 1:
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			float val1=sintdata[tid];
			float val2=myId+s>size?0:sintdata[tid + s];
			sintdata[tid] = val1 + val2;
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	//Step 2: doing downsweep
	sintdata[0] = 0;
	for (unsigned int s = 1;s<blockDim.x / 2; s <<= 1)
	{
		if (tid < s)
		{
			int val1=sintdata[tid];
			int val2=myId+s>size?0:sintdata[tid + s];
			sintdata[tid] = val1+val2;
			if(myId+s<size) sintdata[tid+s] = val1;
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// every thread writes result for this block back to global mem
	result[blockIdx.x] = sintdata[tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	// Two step reduce
	// 1 dimension!!
	int size=numRows * numCols;
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;

	const int STEP1_SHMEM_SIZE=threads * sizeof(float);
	const int STEP2_SHMEM_SIZE=blocks * sizeof(float);

	float *d_intermediate, *d_result;
	checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * blocks));
	checkCudaErrors(cudaMalloc(&d_result, sizeof(float)));

	testToneMapping();

//	std::cout << "Blocks: " << blocks << std::endl;
//	std::cout << "Threads: " << threads << std::endl;

	//TUNING IDEA: MIN AND MAX AT THE SAME TIME

	/* Step 1a: Reduce on d_logLuminance with MIN Operation to get min luminance value*/
	reduceLuminance<<<blocks, threads, STEP1_SHMEM_SIZE>>>(d_logLuminance, false, d_intermediate, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	reduceLuminance<<<1, blocks, STEP2_SHMEM_SIZE>>>(d_intermediate, false, d_result, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_result, sizeof(float), cudaMemcpyDeviceToHost));

	/* Step 1b: Reduce on d_logLuminance with MAX Operation to get max luminance value*/
	reduceLuminance<<<blocks, threads, STEP1_SHMEM_SIZE>>>(d_logLuminance, true, d_intermediate, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	reduceLuminance<<<1, blocks, STEP2_SHMEM_SIZE>>>(d_intermediate, true, d_result, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_result, sizeof(float), cudaMemcpyDeviceToHost));

	// Step 2: calc range
	float range=max_logLum-min_logLum;

//	std::cout << "Min value: " << min_logLum << std::endl;
//	std::cout << "Max value: " << max_logLum << std::endl;
//	std::cout << "Range: " << range << std::endl;
//	std::cout << "Num bins: " << numBins << std::endl;


	//step 3: Histogram
	int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));
	assignHistogram<<<size/threads, threads>>>(d_bins, d_logLuminance, numBins, range, min_logLum);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

//	First 25 bins example
//
//	int h_bins[numBins];
//	checkCudaErrors(cudaMemcpy(&h_bins, d_bins, sizeof(int) * numBins, cudaMemcpyDeviceToHost));
//	for(int i=0;i<25;i++)
//	{
//		std::cout << " - " << i << ": " << h_bins[i] << std::endl;
//	}

	//Last Step: Exclusive Scan on d_bins

	threads=2;
	while(threads<numBins)
		threads<<=1;
	scanHistogram<<<1, threads, sizeof(int) * numBins>>>(d_bins, d_cdf, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());



	//free resources

	cudaFree(d_intermediate);
	cudaFree(d_result);
	cudaFree(d_bins);





  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference cdf on the host by running the            *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated cdf back to the host and calls a function that compares the     *
  * the two and will output the first location they differ.                   *
  * ************************************************************************* */

  /* float *h_logLuminance = new float[numRows * numCols];
  unsigned int *h_cdf   = new unsigned int[numBins];
  unsigned int *h_your_cdf = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_your_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins);

  //compare the results of the CDF
  checkResultsExact(h_cdf, h_your_cdf, numBins);
 
  delete[] h_logLuminance;
  delete[] h_cdf; 
  delete[] h_your_cdf; */
}


void testToneMapping()
{
	float dataToReduce[]={2, 4, 3, 3, 1, 7, 4, 5, 7, 0, 9, 4, 3, 2};
	float min=0, max=0, range=0;
	int numBins=3;
	int threads=2;
	int SIZE=sizeof(dataToReduce)/sizeof(dataToReduce[0]);
	while(threads<SIZE)
		threads<<=1;

	//Step 1

	float *d_in, *d_out;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * SIZE));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_in, dataToReduce, sizeof(float) * SIZE, cudaMemcpyHostToDevice));

	reduceLuminance<<<1, threads, sizeof(float)*SIZE>>>(d_in, false, d_out, SIZE);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&min, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "Minimo: "<<min << std::endl;

	reduceLuminance<<<1, threads, sizeof(float)*SIZE>>>(d_in, true, d_out, SIZE);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&max, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	range=max-min;
	std::cout << "Maximo: "<<max << std::endl;
	std::cout << "Rango: "<<range << std::endl;


	int *d_bins, h_bins[numBins];
	unsigned int *d_cdf, h_cdf[numBins];
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * (numBins)));
	checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * (numBins)));
	checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int) * (numBins)));
	checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * (numBins)));
	assignHistogram<<<1, SIZE>>>(d_bins, d_in, numBins, range, min);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&h_bins, d_bins, sizeof(int)*(numBins), cudaMemcpyDeviceToHost));

	std::cout <<" Entrada ("<< SIZE << "): [ ";
		for(int i=0;i<SIZE;i++)
			std::cout << dataToReduce[i] <<" ";
		std::cout << "]" << std::endl;


	std::cout <<" histograma: [ ";
	for(int i=0;i<numBins;i++)
		std::cout << h_bins[i] <<" ";
	std::cout << "]" << std::endl;


	threads=2;
	while(threads<numBins)
		threads<<=1;
	scanHistogram<<<1, threads, sizeof(int) * numBins>>>(d_bins, d_cdf, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&h_cdf, d_cdf, sizeof(unsigned int)*(numBins), cudaMemcpyDeviceToHost));

	std::cout <<" CDF: [ ";
	for(int i=0;i<numBins;i++)
		std::cout << h_cdf[i] <<" ";
	std::cout << "]" << std::endl;

	cudaFree(d_cdf);
	cudaFree(d_bins);
	cudaFree(d_in);
	cudaFree(d_out);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

