//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>

void testSort();

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__device__ bool isBitByRight(unsigned int number, unsigned int bitIndex)
{
	unsigned int comparator=(1 << bitIndex);
	return (number & comparator) == comparator;
}

__device__ void calcHistogram(unsigned int* const d_inputVals, unsigned int *d_histogram, unsigned int iteration, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;
	//first, shared local
	__shared__ int s_histogram[2];
	if(tid<2)
		s_histogram[tid]=0;
	syncthreads();

	if(myId<numElems)
	{

		bool isOne=isBitByRight(d_inputVals[myId], iteration);
		atomicAdd(&(s_histogram[isOne?1:0]), 1);
		syncthreads();

		//add to global count
		if(tid<2) {
			atomicAdd(&(d_histogram[tid]), s_histogram[tid]);
		}
		syncthreads();
	}
}

__device__ void compact(unsigned int* const d_inputVals,
        unsigned int* const d_inputPos,
        unsigned int const indexSecondArray,
        const size_t numElems, unsigned int iteration,
        unsigned int* d_intermediate, unsigned int* d_scatterAddress)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	//int tid  = threadIdx.x;

	if(myId<numElems)
	{
		//Step 1: evaluate value 0
		//first, evaluate predicate
		bool isOne=isBitByRight(d_inputVals[myId], iteration);
		d_intermediate[myId] = isOne?0:1;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (int step = 1; step<numElems;step <<= 1)
		{
			if (myId >=step)
			{
				d_intermediate[myId] += d_intermediate[myId - step];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}
		//copy yo final scatter address
		if(!isOne)
			d_scatterAddress[myId]=d_intermediate[myId]-1;

		//step 2: evaluate value 1
		d_intermediate[myId] = isOne?1:0;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (int step = 1; step<numElems;step <<= 1)
		{
			if (myId >=step)
			{
				d_intermediate[myId] += d_intermediate[myId - step];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}

		//copy yo final scatter address
		if(isOne)
			d_scatterAddress[myId]=indexSecondArray + d_intermediate[myId]-1;
		__syncthreads();
	}

}

__device__ void scatter(unsigned int* const d_inputVals, unsigned int* const d_inputPos,
		unsigned int* const d_outputVals, unsigned int* const d_outputPos,
		unsigned int* d_scatterAddress, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if(myId<numElems)
	{
		unsigned int address=d_scatterAddress[myId];
		if(address<numElems)
		{
			d_outputVals[address]=d_inputVals[myId];
			d_outputPos[address]=d_inputPos[myId];
		}
	}
}

__global__ void sortPixels(unsigned int* const d_inputVals,
        unsigned int* const d_inputPos,
        unsigned int* const d_outputVals,
        unsigned int* const d_outputPos,
        const size_t numElems, unsigned int iteration, unsigned int *d_histogram,
        unsigned int* d_intermediate, unsigned int* d_scatterAddress)
{
	//each thread is realted to one element
	//Step 1: Histogram
	calcHistogram(d_inputVals, d_histogram, iteration, numElems);

	//Step 2: Compact t calc scatter address
	compact(d_inputVals, d_inputPos, d_histogram[0], numElems, iteration, d_intermediate, d_scatterAddress);

	//step 3: scatter
	scatter(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scatterAddress, numElems);


}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
  * the input values when implementing your radix sort.                       *
  *                                                                           *
  * This code performs the reference radix sort on the host and compares your *
  * sorted values to the reference.                                           *
  *                                                                           *
  * Thrust containers are used for copying memory from the GPU                *
  * ************************************************************************* */
  
  /* thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
                                                thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
                                               thrust::device_ptr<unsigned int>(d_inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems);
  */

	testSort();
   

	unsigned int maxNumThreads=1024;
	unsigned int blocks=ceil(((double)numElems / (double)maxNumThreads));

	unsigned int numIterations=sizeof(unsigned int)*8; //one iteration per bit

//	std::cout << "Threads: "<<maxNumThreads<<std::endl;
//	std::cout << "Blocks: "<<blocks<<std::endl;
//	std::cout << "Iterations: "<<numIterations<<std::endl;

	unsigned int *d_histogram, *d_intermediate, *d_scatterAddress;
	checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int)*2));
	checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_scatterAddress, sizeof(unsigned int)*numElems));

	for(unsigned int i=0;i<numIterations;i++)
	{
		checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int)*2));
		checkCudaErrors(cudaMemset(d_intermediate, 0, sizeof(unsigned int)*numElems));
		sortPixels<<<blocks, maxNumThreads>>>(i==0?d_inputVals:d_outputVals, i==0?d_inputPos:d_outputPos, d_outputVals, d_outputPos, numElems, i, d_histogram, d_intermediate, d_scatterAddress);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//		unsigned int h_histogram[2];
		//		cudaMemcpy(&h_histogram, d_histogram, sizeof(unsigned int)*2, cudaMemcpyDeviceToHost);
		//		std::cout << "Iteration "<<i<<":"<<std::endl;
		//		std::cout << " - 0: "<<h_histogram[0]<<std::endl;
		//		std::cout << " - 1: "<<h_histogram[1]<<std::endl;
		//		std::cout << " - total: "<<(h_histogram[0]+h_histogram[1])<<" (should be "<<numElems<<")"<<std::endl;

//		unsigned int h_scatterAddress[numElems];
//		cudaMemcpy(&h_scatterAddress, d_scatterAddress, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToHost);
//		for(int j=0;j<10;j++)
//			std::cout << " Intermediante "<<j<<":"<< h_scatterAddress[j] <<" of " <<numElems <<std::endl;

	}

	checkCudaErrors(cudaFree(d_scatterAddress));
	checkCudaErrors(cudaFree(d_intermediate));
	checkCudaErrors(cudaFree(d_histogram));


  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/

  /*
  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
                                                     thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
                                                    thrust::device_ptr<unsigned int>(d_outputPos) + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);
  */
}

void print(char *message, unsigned int *data, unsigned int numElems)
{
	std::cout << message <<": [";
	for(unsigned int i=0;i<numElems;i++)
		std::cout << " " << data[i] << " ";
	std::cout <<"]"<<std::endl;
}

void testSort()
{
	const size_t numElems = 13;
	unsigned int h_data[]={32, 5, 23, 54, 54, 32, 12, 34, 4, 4, 123, 213, 43};
	unsigned int h_dataVals[]={32, 5, 23, 54, 54, 32, 12, 34, 4, 4, 123, 213, 43};
	int bytesSize=sizeof(unsigned int)*numElems;
	unsigned int *h_outputPos=(unsigned int *)malloc(sizeof(unsigned int)*numElems);
	unsigned int *h_outputVals=(unsigned int *)malloc(sizeof(unsigned int)*numElems);

	unsigned int numIterations=10;

	unsigned int *d_data, *d_dataVals, *d_outputPos, *d_outputVals;
	cudaMalloc(&d_data, bytesSize);
	cudaMalloc(&d_dataVals, bytesSize);
	cudaMalloc(&d_outputPos, bytesSize);
	cudaMalloc(&d_outputVals, bytesSize);
	cudaMemcpy(d_data, h_data, bytesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataVals, h_dataVals, bytesSize, cudaMemcpyHostToDevice);



	unsigned int *d_histogram, *d_intermediate, *d_scatterAddress;
	checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int)*2));
	checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_scatterAddress, sizeof(unsigned int)*numElems));

	print("Original data", h_data, numElems);
	//print("Original values", h_dataVals, numElems);

	for(unsigned int i=0;i<numIterations;i++)
	{
		checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int)*2));
		checkCudaErrors(cudaMemset(d_intermediate, 0, sizeof(unsigned int)*numElems));
		sortPixels<<<2, 7>>>(i==0?d_dataVals:d_outputVals, i==0?d_data:d_outputPos, d_outputVals, d_outputPos, numElems, i, d_histogram, d_intermediate, d_scatterAddress);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		unsigned int h_scatterAddress[numElems];
		cudaMemcpy(&h_scatterAddress, d_scatterAddress, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToHost);
		unsigned int h_histogram[2];
		cudaMemcpy(&h_histogram, d_histogram, sizeof(unsigned int)*2, cudaMemcpyDeviceToHost);
		std::cout << " Intermediate "<<i<<": [";
		for(unsigned int j=0;j<numElems;j++)
			std::cout << " " << h_scatterAddress[j] <<" ";
		std::cout << "] (0="<<h_histogram[0]<<", 1="<<h_histogram[1]<<")"<<std::endl;
		cudaMemcpy(h_outputPos, d_outputPos, bytesSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputVals, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
		print("   - Output data", h_outputPos, numElems);
		//print("Output values", h_outputVals, numElems);
	}
	print("Output data", h_outputPos, numElems);



	free(h_outputVals);
	free(h_outputPos);
	checkCudaErrors(cudaFree(d_scatterAddress));
	checkCudaErrors(cudaFree(d_intermediate));
	checkCudaErrors(cudaFree(d_histogram));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_dataVals));
	checkCudaErrors(cudaFree(d_outputPos));
	checkCudaErrors(cudaFree(d_outputVals));
}

