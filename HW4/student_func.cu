//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <thrust/host_vector.h>

void testSort();


std::string format(int number)
{
	std::ostringstream oss;
	oss << std::setfill(' ') << std::setw(3) << number;
	return oss.str();
}

void print(char *message, unsigned int *data, unsigned int numElems)
{
	std::cout << "\033[1;32m" << message <<"\033[0m: [";
	for(unsigned int i=0;i<numElems;i++)
	{
		bool isOk=i==0 || data[i-1]<=data[i];
		std::cout << " " <<(isOk?"":"\033[1;31m") << format(data[i]) <<(isOk?"":"\033[0m") << " ";
	}
	std::cout <<"]"<<std::endl;
}

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

bool h_isBitByRight(unsigned int number, unsigned int bitIndex)
{
	unsigned int comparator=(1 << bitIndex);
	return (number & comparator) == comparator;
}

__device__ bool isBitByRight(unsigned int number, unsigned int bitIndex)
{
	unsigned int comparator=(1 << bitIndex);
	return (number & comparator) == comparator;
}

__device__ void calcHistogram(unsigned int const s_dataVals[], unsigned int s_histogram[], unsigned int iteration, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	if(myId<numElems)
	{
		bool isOne=isBitByRight(s_dataVals[tid], iteration);
		atomicAdd(&(s_histogram[isOne?1:0]), 1);
	}

	__syncthreads();
}

__device__ void compact(unsigned int const *s_dataVals,
        unsigned int const indexSecondArray,
        const size_t numElems, unsigned int iteration,
        unsigned int* s_intermediate, unsigned int *s_scatterAddress)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	if(myId<numElems)
	{
		//Step 1: evaluate value 0
		//first, evaluate predicate
		bool isOne=isBitByRight(s_dataVals[tid], iteration);
		s_intermediate[tid] = isOne?0:1;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (int step = 1; step<blockDim.x;step <<= 1)
		{
			if (tid >=step)
			{
				s_intermediate[tid] += s_intermediate[tid - step];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}
		//copy yo final scatter address
		if(!isOne)
			s_scatterAddress[tid]=s_intermediate[tid]-1;

		//step 2: evaluate value 1
		s_intermediate[tid] = isOne?1:0;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (int step = 1; step<blockDim.x;step <<= 1)
		{
			if (tid >=step)
			{
				s_intermediate[tid] += s_intermediate[tid - step];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}

		//copy yo final scatter address
		if(isOne)
			s_scatterAddress[tid]=indexSecondArray + s_intermediate[tid]-1;


		__syncthreads();

	}

}

__device__ void scatter(unsigned int* const s_inputVals, unsigned int* s_outputVals,
		unsigned int* s_scatterAddress, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	if(myId<numElems)
	{
		unsigned int address=s_scatterAddress[tid];
		s_outputVals[address]=s_inputVals[tid];
		//s_outputVals[tid]=address;
	}
}

__global__ void sortPixelsSegmented(unsigned int* const d_inputVals, unsigned int* const d_inputPos,
        unsigned int* const d_outputVals, unsigned int* const d_outputPos,
        const size_t numElems)
{
	const unsigned int numIterations=sizeof(unsigned int) * 8;

	// shared memory data
	__shared__ unsigned int s_histogram[2];
	extern __shared__ unsigned int s_dataCopy[]; //vals and pos concatenated

	unsigned int *s_dataVals=s_dataCopy;
	unsigned int *s_dataPos=s_dataCopy + blockDim.x;
	unsigned int *s_intermediate=s_dataPos + blockDim.x;
	unsigned int *s_scatterAddress=s_intermediate + blockDim.x;

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	//copy data to shared memory
	if(myId>numElems) return;
	s_dataVals[tid] = d_inputVals[myId];
	s_dataPos[tid] = d_inputPos[myId];

	for(unsigned int iteration=0;iteration<numIterations;iteration++)
	{
		if(tid<2)
			s_histogram[tid]=0;
		__syncthreads();

		//each thread is realted to one element
		//Step 1: Histogram
		calcHistogram(s_dataVals, s_histogram, iteration, numElems);
		__syncthreads();

		//Step 2: Compact t calc scatter address
		compact(s_dataVals, s_histogram[0], numElems, iteration, s_intermediate, s_scatterAddress);
		__syncthreads();

		//step 3: scatter
		scatter(s_dataVals, s_intermediate, s_scatterAddress, numElems);
		__syncthreads();
		s_dataVals[tid]=s_intermediate[tid];
		__syncthreads();

		scatter(s_dataPos, s_intermediate, s_scatterAddress, numElems);
		__syncthreads();
		s_dataPos[tid]=s_intermediate[tid];
		__syncthreads();

	}


	//copy data back from shared memory
	d_outputVals[myId]=s_dataVals[tid];
	d_outputPos[myId]=s_dataPos[tid];
	__syncthreads();
}

// Busqueda dicotomica
__device__ int searchInArray(unsigned const int value, unsigned int *s_dataVals, int const idxStartSearch,
		int const size, const size_t numElems, const bool isSecondArray)
{
	int index=0, searchIndex=idxStartSearch;


	for(int split=(size>>1);;split=(split>>1))
	{
		if((searchIndex + split) > numElems)
		{
			if(split==0)
				break;
			else
				continue;
		}

		unsigned int compareValue=s_dataVals[searchIndex  + split];

		if(value > compareValue)
		{
			searchIndex+=split;
			index+=(split==0?1:split);
		}
		else if(isSecondArray and (value - compareValue==0))
		{
			index+=1;
		}
		if(split==0) break; //hemos llegado al final
	}
	return index;
}

__global__ void joinSegmented(unsigned int *s_dataVals, unsigned int *s_dataPos, const size_t numElems, const int i)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	//int tid  = threadIdx.x;

	if(myId>=numElems) return;

	//int maxSize=blockDim.x *gridDim.x; //asumimos que es potencia de 2
	int currentSize=blockDim.x << i;

	//indices de los arrays que corresponden a este elemento
	int idxFirstArray=floorf(myId/(currentSize*2)) * currentSize * 2;
	int idxSecondArray=idxFirstArray + currentSize;

	//TODO: Usar shared memory
	unsigned int myValue=s_dataVals[myId];
	unsigned int myPos=s_dataPos[myId];
	syncthreads();

	bool isSecondArray=(myId>=idxSecondArray);
	int idxCurrentArray=(isSecondArray?(myId - idxSecondArray):(myId - idxFirstArray));
	int idxOtherArray=(idxSecondArray>numElems)?0:searchInArray(myValue, s_dataVals, (isSecondArray?idxFirstArray: idxSecondArray), currentSize, numElems, isSecondArray);
	int newIndex=idxCurrentArray+idxOtherArray;
	syncthreads(); //esperamos a que todos los threads sepan a donde deben ir con su dato

	//establezco el nuevo dato
	s_dataVals[idxFirstArray + newIndex] = myValue;
	s_dataPos[idxFirstArray + newIndex] = myPos;
	syncthreads();
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

//  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
//                                                thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
//  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
//                                               thrust::device_ptr<unsigned int>(d_inputPos) + numElems);
//
//  thrust::host_vector<unsigned int> h_outputVals(numElems);
//  thrust::host_vector<unsigned int> h_outputPos(numElems);
//
//  reference_calculation(&h_inputVals[0], &h_inputPos[0],
//                        &h_outputVals[0], &h_outputPos[0],
//                        numElems);


	testSort();

//
//	const unsigned int NUM_THREADS=50;
//	unsigned int NUM_BLOCKS=ceil(((double)numElems / (double)NUM_THREADS));
//	unsigned int powof2 = 1;
//	// Double powof2 until >= val
//	while( powof2 < NUM_BLOCKS ) powof2 <<= 1;
//	NUM_BLOCKS=powof2;
//
//
//	const unsigned int BYTES_PER_ARRAY = NUM_THREADS * sizeof(unsigned int);
//
//	std::cout << "NUM_THREADS: "<<NUM_THREADS<<std::endl;
//	std::cout << "NUM_BLOCKS: "<<NUM_BLOCKS<<std::endl;
//	std::cout << "numElems: "<<numElems<<std::endl;
//	std::cout << "BYTES_PER_ARRAY: "<<BYTES_PER_ARRAY<<std::endl;
//
//	int bytesSize=sizeof(unsigned int)*numElems;
//	unsigned int *h_outputVals2=(unsigned int *)malloc(bytesSize);
//	unsigned int *h_inputVals2=(unsigned int *)malloc(bytesSize);
//	sortPixelsSegmented<<<NUM_BLOCKS, NUM_THREADS, 4 * BYTES_PER_ARRAY>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
//	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//	cudaMemcpy(h_inputVals2, d_inputVals, bytesSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_outputVals2, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
//
//	print("Orig", h_inputVals2, NUM_THREADS);
//	print("Data", h_outputVals2, NUM_THREADS);
//
//
//	int numIterations=ceil(log2((float)NUM_BLOCKS));
//
//	std::cout << "numIterations: "<<numIterations<<std::endl;
//
//	for(int i=0;i<numIterations;i++)
//	{
//		joinSegmented<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_outputPos, numElems, i);
//		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//	}
//
//	free(h_inputVals2);
//	free(h_outputVals2);


  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/
//
//
//  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
//                                                     thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
//  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
//                                                    thrust::device_ptr<unsigned int>(d_outputPos) + numElems);
//
//  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
//  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);

}



void testSort()
{
	std::cout<<"--------------------------"<<std::endl;

//	const size_t numElems = 13;
//	unsigned int h_data[]={32, 5, 23, 54, 55, 33, 12, 34, 4, 6, 123, 213, 44};
//	unsigned int h_dataVals[]={32, 5, 23, 54, 55, 33, 12, 34, 4, 6, 123, 213, 44};
	const size_t numElems = 50;
	unsigned int h_data[] ={1040146228, 1040124922, 1040087966, 1040044167, 1040060259, 1040191523, 1040281730, 1040321287,
						   1040439347, 1040511719, 1040534712, 1040482053, 1040379127, 1040260380, 1040082964, 1039836996,
						   1039601849, 1039367071, 1039218350, 1039287569, 1039483561, 1039791413, 1040074985, 1040079855,
						   1039744906, 1039356991, 1039396067, 1039691868, 1040032759, 1040168105, 1040144074, 1040102826,
						   1040159349, 1040218642, 1040290902, 1040404583, 1040492144, 1040470157 ,1040405164 ,1040378600,
						   1040259132 ,1040168513 ,1040126140 ,1040122880 ,1040122680 ,1040122331 ,1040114162 ,1040092632,
						   1040026226 ,1039921833 };
	unsigned int h_dataVals[] ={1040146228, 1040124922, 1040087966, 1040044167, 1040060259, 1040191523, 1040281730, 1040321287,
						   1040439347, 1040511719, 1040534712, 1040482053, 1040379127, 1040260380, 1040082964, 1039836996,
						   1039601849, 1039367071, 1039218350, 1039287569, 1039483561, 1039791413, 1040074985, 1040079855,
						   1039744906, 1039356991, 1039396067, 1039691868, 1040032759, 1040168105, 1040144074, 1040102826,
						   1040159349, 1040218642, 1040290902, 1040404583, 1040492144, 1040470157 ,1040405164 ,1040378600,
						   1040259132 ,1040168513 ,1040126140 ,1040122880 ,1040122680 ,1040122331 ,1040114162 ,1040092632,
						   1040026226 ,1039921833 };
	int bytesSize=sizeof(unsigned int)*numElems;
	unsigned int *h_outputPos=(unsigned int *)malloc(sizeof(unsigned int)*numElems);
	unsigned int *h_outputVals=(unsigned int *)malloc(sizeof(unsigned int)*numElems);

	unsigned int *d_data, *d_dataVals, *d_outputPos, *d_outputVals;
	cudaMalloc(&d_data, bytesSize);
	cudaMalloc(&d_dataVals, bytesSize);
	cudaMalloc(&d_outputPos, bytesSize);
	cudaMalloc(&d_outputVals, bytesSize);
	cudaMemcpy(d_data, h_data, bytesSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataVals, h_dataVals, bytesSize, cudaMemcpyHostToDevice);

	print("- Original data", h_data, numElems);
	//print_is_zero("Original ZERO", h_data, numElems, 0);
	print("Original values", h_dataVals, numElems);
	std::cout<<"--------------------------"<<std::endl;

	const unsigned int NUM_BLOCKS=16;
	const unsigned int NUM_THREADS=4;
	const unsigned int BYTES_PER_ARRAY = NUM_BLOCKS * NUM_THREADS * sizeof(unsigned int);

	//MERGE SORT: Step 1, sort inside block
	sortPixelsSegmented<<<NUM_BLOCKS, NUM_THREADS, 4 * BYTES_PER_ARRAY>>>(d_dataVals, d_data, d_outputVals, d_outputPos, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	cudaMemcpy(h_outputPos, d_outputPos, bytesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_outputVals, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
	print("S - Output pos.", h_outputPos, numElems);
	print("S - Output data", h_outputVals, numElems);

	//MERGE SORT: Step 2, join segments
	int numIterations=ceil(log2((float)NUM_BLOCKS));
	for(int i=0;i<numIterations;i++)
	{
		std::cout<<"--------------------------"<<std::endl;
		joinSegmented<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_outputPos, numElems, i);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		cudaMemcpy(h_outputPos, d_outputPos, bytesSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputVals, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
		char iter[20];
		sprintf(iter, "%d - Output pos.", i);
		print(iter, h_outputPos, numElems);
		sprintf(iter, "%d - Output data", i);
		print(iter, h_outputVals, numElems);
	}

	std::cout<<"--------------------------"<<std::endl;

	free(h_outputVals);
	free(h_outputPos);
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_dataVals));
	checkCudaErrors(cudaFree(d_outputPos));
	checkCudaErrors(cudaFree(d_outputVals));
}

