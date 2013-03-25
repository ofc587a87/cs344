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

//void testSort();


std::string format(unsigned int number)
{
	std::ostringstream oss;
	oss << std::setfill(' ') << std::setw(3) << number;
	return oss.str();
}

void print(char *message, unsigned int *data, unsigned int fromElem, unsigned int numElems)
{
	std::cout << "\033[1;32m" << message <<"\033[0m: [";
	for(unsigned int i=fromElem;i<numElems;i++)
	{
		bool isOk=i==0 || data[i-1]<=data[i];
		std::cout << " " <<(isOk?"":"\033[1;31m") << format(data[i]) <<(isOk?"":"\033[0m") << " ";
	}
	std::cout <<"]"<<std::endl;
}

void printFilter(char *message, unsigned int *data, unsigned int numElems, unsigned int maxValue)
{
	std::cout << "\033[1;32m" << message <<"\033[0m: [";
	for(unsigned int i=0;i<numElems;i++)
	{
		if(data[i]>maxValue)
		std::cout << " \033[1;31m" << format(data[i]) <<"\033[0m" << " (POS: "<<i<<")";
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

__device__ bool isBitByRight(unsigned int number, unsigned int bitIndex)
{
	unsigned int comparator=(1 << bitIndex);
	return (number & comparator) == comparator;
}

__device__ int calcHistogram(unsigned int const *s_dataVals, unsigned int iteration, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;
	__shared__ int s_histogram;
	if(tid==0)
		s_histogram=0;
	__syncthreads();

	if(myId<numElems)
	{
		if(!isBitByRight(s_dataVals[tid], iteration))
			atomicAdd(&s_histogram, 1); //calculamos los ceros para saber donde empezaran los unos
	}
	__syncthreads();

	return s_histogram;
}

__device__ void compact(unsigned int const *s_dataVals, unsigned int const indexSecondArray,
        const size_t numElems, unsigned int iteration,
        unsigned int* s_intermediate, unsigned int *d_scatterAddress)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid  = threadIdx.x;

	if(myId<numElems)
	{
		//Step 1: evaluate value 0
		//first, evaluate predicate
		bool isOne=isBitByRight(s_dataVals[tid], iteration);
		s_intermediate[tid] = isOne?0:1;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (unsigned int step = 1; step<blockDim.x;step <<= 1)
		{
			unsigned int value=s_intermediate[tid - step];
			__syncthreads();
			if (tid >=step)
				s_intermediate[tid] += value;

			__syncthreads();        // make sure all adds at one stage are done!
		}
		//copy yo final scatter address
		if(!isOne)
			d_scatterAddress[myId]=s_intermediate[tid]-1;
		__syncthreads();

		//step 2: evaluate value 1
		s_intermediate[tid] = isOne?1:0;
		__syncthreads();

		//scan using predicate to compute scatter address
		//Step hillis / Steele
		for (int step = 1; step<blockDim.x;step <<= 1)
		{
			unsigned int value=s_intermediate[tid - step];
			__syncthreads();
			if (tid >=step)
				s_intermediate[tid] +=value;
			__syncthreads();        // make sure all adds at one stage are done!
		}

		//copy to final scatter address
		if(isOne)
			d_scatterAddress[myId]=indexSecondArray + s_intermediate[tid]-1;

		__syncthreads();

	}
}

__global__ void radixBlock(unsigned int *d_dataVals, unsigned int *d_dataPos,
						   unsigned int *d_histogram, unsigned int *d_scatterAddress,
						   const size_t numElems, int iteration)
{
	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid  = threadIdx.x;

	//copia datos del bloque a ordenar
	extern __shared__ unsigned int s_dataVals[];
	unsigned int *s_intermediate=s_dataVals+blockDim.x;
	s_dataVals[tid]=d_dataVals[myId];
	s_intermediate[tid]=0;
	__syncthreads();

	//calcula el indice con bit 1 para el histograma
	int histogramIndex=calcHistogram(s_dataVals, iteration, numElems);
	__syncthreads();

	if(tid==0)
		d_histogram[blockIdx.x]=histogramIndex;

	compact(s_dataVals, histogramIndex, numElems, iteration, s_intermediate, d_scatterAddress);
}

__global__ void calcScatterAddress(unsigned int *d_scaterAddress, unsigned int *d_histogram,
		                           const unsigned int totalHistogram, const size_t numElems,
		                           unsigned const int BLOCK_SIZE, unsigned const int maxRealBlocks)
{
	unsigned int tid=threadIdx.x;

	extern __shared__ unsigned s_data[];
	unsigned int *s_histogram_ZERO=s_data;
	unsigned int *s_histogram_ONE=s_data + blockDim.x;

	//limit histogram size
	int blockSize=BLOCK_SIZE;
	if(tid==maxRealBlocks)
	{
		blockSize=numElems % BLOCK_SIZE;
	}


	//copy histogram to shared memory
	//Whe want an exclusive Scan, so copy data SHIFTED (first zero, others tid-1)
	int barrier=d_histogram[tid];
	unsigned int localData=tid==0?0:(d_histogram[tid-1]);
	s_histogram_ZERO[tid]=tid==0?0:localData-1; //histogram is count elements, but address starts a zero, so quit 1
	s_histogram_ONE[tid] = tid==0?0:blockSize - localData; //ONEs doesn't need to be shifted
	if(tid>maxRealBlocks)
		s_histogram_ONE[tid]=0;
	__syncthreads();

	//Inclusive scan of both arrays (but it's exclusive as the data is sifted)
	//Step hillis / Steele
	for (int step = 1; step<BLOCK_SIZE;step <<= 1)
	{
		unsigned int val_ZERO=0, val_ONE=0;
		if (tid >=step)
		{
			val_ZERO=s_histogram_ZERO[tid - step];
			val_ONE=s_histogram_ONE[tid - step];
		}
		__syncthreads();

		if (tid >=step)
		{
			s_histogram_ZERO[tid] += val_ZERO;
			s_histogram_ONE[tid] += val_ONE;
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	s_histogram_ONE[tid]+=totalHistogram - barrier;
	d_histogram[tid]=s_histogram_ONE[tid];
	__syncthreads();

	//calc global scatter address
	//each thread modifys one block
	int from=tid * BLOCK_SIZE;

	for(int i=0;i<BLOCK_SIZE;i++)
	{
		if((from+i)>numElems)
			break;

		unsigned int value=d_scaterAddress[from + i];
		if(value < barrier)
			value+= s_histogram_ZERO[tid];
		else
			value+=s_histogram_ONE[tid];
		d_scaterAddress[from + i]=value;
		//d_scaterAddress[from + i]=localData;
	}
}

__global__ void scater(unsigned int *d_dataVals, unsigned int *d_dataPos,
		               unsigned int *d_outputVals, unsigned int *d_outputPos,
					   unsigned int *d_scatterAddress, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if(myId<numElems)
	{
		unsigned int address=d_scatterAddress[myId];
		if(address<numElems)
		{
			d_outputVals[address]=d_dataVals[myId];
			d_outputPos[address]=d_dataPos[myId];
		}
	}

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


//	testSort();
	const unsigned int NUM_THREADS=1024;
	unsigned int MAX_REAL_BLOCKS=ceil(((double)numElems / (double)NUM_THREADS));
	const unsigned int BYTES_PER_ARRAY=NUM_THREADS * sizeof(unsigned int);

	unsigned int powof2 = 1;
	// Double powof2 until >= val
	while( powof2 < MAX_REAL_BLOCKS ) powof2 <<= 1;
	unsigned int NUM_BLOCKS=powof2;


	//const unsigned int BYTES_PER_ARRAY = NUM_THREADS * sizeof(unsigned int);

	std::cout << "NUM_THREADS: "<<NUM_THREADS<<std::endl;
	std::cout << "NUM_BLOCKS: "<<NUM_BLOCKS<<std::endl;
	std::cout << "numElems: "<<numElems<<std::endl;
	std::cout << "BYTES_PER_ARRAY: "<<BYTES_PER_ARRAY<<std::endl;

	unsigned int *d_histogram, *d_scatterAddress, *d_inputTempVals, *d_inputTempPos;
	cudaMalloc(&d_scatterAddress, sizeof(unsigned int) * numElems);
	cudaMalloc(&d_histogram, sizeof(unsigned int) * NUM_BLOCKS);
	cudaMalloc(&d_inputTempVals, sizeof(unsigned int) * numElems);
	cudaMalloc(&d_inputTempPos, sizeof(unsigned int) * numElems);

	int bytesSize=sizeof(unsigned int)*numElems;
	cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);

	const unsigned int numIterations=sizeof(unsigned int) * 8;
	for(unsigned int iteration=0;iteration<numIterations;iteration++)
	{
		cudaMemcpy(d_inputTempVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_inputTempPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);

		std::cout << "Iteration "<<iteration<<" of "<<numIterations<<std::endl;
		//Step 1: Histogram and order local block
		cudaMemset(d_histogram, 0, sizeof(unsigned int) * NUM_BLOCKS);
		cudaMemset(d_scatterAddress, 0, sizeof(unsigned int) * numElems);
		radixBlock<<<NUM_BLOCKS, NUM_THREADS, BYTES_PER_ARRAY * 2>>>(d_inputTempVals, d_inputTempPos, d_histogram, d_scatterAddress, numElems,  iteration);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//fast calc: total histogram
		unsigned int totalHistogram=0;
		unsigned int h_histogram[NUM_BLOCKS];
		cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
		for(unsigned int i=0;i<NUM_BLOCKS;i++)
			totalHistogram+=h_histogram[i];

//		print("HISTOGRAM", h_histogram, 0, MAX_REAL_BLOCKS);

		unsigned int h_scater[numElems];
		cudaMemcpy(h_scater, d_scatterAddress, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost);
		std::cout << " TOTAL HISTOGRAM: " << totalHistogram<<std::endl;
//		print("SCATER BEFORE:  ", h_scater, 220000, numElems);
		printFilter("SCATER ERRORS BEFORE:  ", h_scater, numElems, NUM_THREADS-1);

		//Steo 2: calc scater address
		calcScatterAddress<<<1, NUM_BLOCKS, NUM_BLOCKS * sizeof(unsigned int) * 2>>>(d_scatterAddress, d_histogram, totalHistogram, numElems, NUM_THREADS, MAX_REAL_BLOCKS);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

//		cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
//		print("HISTOGRAM SCAN:", h_histogram, 0, MAX_REAL_BLOCKS);


		cudaMemcpy(h_scater, d_scatterAddress, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost);
//		print("SCATER AFTER:  ", h_scater, 220000, numElems);
		printFilter("\n\n\nSCATER ERRORS:  ", h_scater, numElems, numElems-1);


		//step 3: scater!
		scater<<<NUM_BLOCKS, NUM_THREADS>>>(d_inputTempVals, d_inputTempPos, d_outputVals, d_outputPos, d_scatterAddress, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	}

	unsigned int *h_outputVals2=(unsigned int *)malloc(bytesSize);
	unsigned int *h_inputVals2=(unsigned int *)malloc(bytesSize);

	cudaMemcpy(h_inputVals2, d_inputVals, bytesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_outputVals2, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);

	std::cout << "ORIG DATA LOCATION: " << d_inputVals<<std::endl;
	std::cout << "OUTP DATA LOCATION: " << d_outputVals<<std::endl;

	print("Orig", h_inputVals2, 0, NUM_THREADS);
	print("Data", h_outputVals2, 0, NUM_THREADS);


/*	int numIterations=ceil(log2((float)NUM_BLOCKS));

	std::cout << "numIterations: "<<numIterations<<std::endl;

	for(int i=0;i<numIterations;i++)
	{
		joinSegmented<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_outputPos, numElems, i);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}*/


	free(h_inputVals2);
	free(h_outputVals2);
	cudaFree(d_histogram);
	cudaFree(d_scatterAddress);
	cudaFree(d_inputTempVals);
	cudaFree(d_inputTempPos);


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



//void testSort()
//{
//	std::cout<<"--------------------------"<<std::endl;
//
////	const size_t numElems = 13;
////	unsigned int h_data[]={32, 5, 23, 54, 55, 33, 12, 34, 4, 6, 123, 213, 44};
////	unsigned int h_dataVals[]={32, 5, 23, 54, 55, 33, 12, 34, 4, 6, 123, 213, 44};
//	const size_t numElems = 50;
//	unsigned int h_data[] ={1040146228, 1040124922, 1040087966, 1040044167, 1040060259, 1040191523, 1040281730, 1040321287,
//						   1040439347, 1040511719, 1040534712, 1040482053, 1040379127, 1040260380, 1040082964, 1039836996,
//						   1039601849, 1039367071, 1039218350, 1039287569, 1039483561, 1039791413, 1040074985, 1040079855,
//						   1039744906, 1039356991, 1039396067, 1039691868, 1040032759, 1040168105, 1040144074, 1040102826,
//						   1040159349, 1040218642, 1040290902, 1040404583, 1040492144, 1040470157 ,1040405164 ,1040378600,
//						   1040259132 ,1040168513 ,1040126140 ,1040122880 ,1040122680 ,1040122331 ,1040114162 ,1040092632,
//						   1040026226 ,1039921833 };
//	unsigned int h_dataVals[] ={1040146228, 1040124922, 1040087966, 1040044167, 1040060259, 1040191523, 1040281730, 1040321287,
//						   1040439347, 1040511719, 1040534712, 1040482053, 1040379127, 1040260380, 1040082964, 1039836996,
//						   1039601849, 1039367071, 1039218350, 1039287569, 1039483561, 1039791413, 1040074985, 1040079855,
//						   1039744906, 1039356991, 1039396067, 1039691868, 1040032759, 1040168105, 1040144074, 1040102826,
//						   1040159349, 1040218642, 1040290902, 1040404583, 1040492144, 1040470157 ,1040405164 ,1040378600,
//						   1040259132 ,1040168513 ,1040126140 ,1040122880 ,1040122680 ,1040122331 ,1040114162 ,1040092632,
//						   1040026226 ,1039921833 };
//	int bytesSize=sizeof(unsigned int)*numElems;
//	unsigned int *h_outputPos=(unsigned int *)malloc(sizeof(unsigned int)*numElems);
//	unsigned int *h_outputVals=(unsigned int *)malloc(sizeof(unsigned int)*numElems);
//
//	unsigned int *d_data, *d_dataVals, *d_outputPos, *d_outputVals;
//	cudaMalloc(&d_data, bytesSize);
//	cudaMalloc(&d_dataVals, bytesSize);
//	cudaMalloc(&d_outputPos, bytesSize);
//	cudaMalloc(&d_outputVals, bytesSize);
//	cudaMemcpy(d_data, h_data, bytesSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_dataVals, h_dataVals, bytesSize, cudaMemcpyHostToDevice);
//
//	print("- Original data", h_data, numElems);
//	//print_is_zero("Original ZERO", h_data, numElems, 0);
//	print("Original values", h_dataVals, numElems);
//	std::cout<<"--------------------------"<<std::endl;
//
//	const unsigned int NUM_BLOCKS=16;
//	const unsigned int NUM_THREADS=4;
//	const unsigned int BYTES_PER_ARRAY = NUM_BLOCKS * NUM_THREADS * sizeof(unsigned int);
//
//	//MERGE SORT: Step 1, sort inside block
//	sortPixelsSegmented<<<NUM_BLOCKS, NUM_THREADS, 4 * BYTES_PER_ARRAY>>>(d_dataVals, d_data, d_outputVals, d_outputPos, numElems);
//	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//	cudaMemcpy(h_outputPos, d_outputPos, bytesSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_outputVals, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
//	print("S - Output pos.", h_outputPos, numElems);
//	print("S - Output data", h_outputVals, numElems);
//
//	//MERGE SORT: Step 2, join segments
//	int numIterations=ceil(log2((float)NUM_BLOCKS));
//	for(int i=0;i<numIterations;i++)
//	{
//		std::cout<<"--------------------------"<<std::endl;
//		joinSegmented<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_outputPos, numElems, i);
//		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//
//		cudaMemcpy(h_outputPos, d_outputPos, bytesSize, cudaMemcpyDeviceToHost);
//		cudaMemcpy(h_outputVals, d_outputVals, bytesSize, cudaMemcpyDeviceToHost);
//		char iter[20];
//		sprintf(iter, "%d - Output pos.", i);
//		print(iter, h_outputPos, numElems);
//		sprintf(iter, "%d - Output data", i);
//		print(iter, h_outputVals, numElems);
//	}
//
//	std::cout<<"--------------------------"<<std::endl;
//
//	free(h_outputVals);
//	free(h_outputPos);
//	checkCudaErrors(cudaFree(d_data));
//	checkCudaErrors(cudaFree(d_dataVals));
//	checkCudaErrors(cudaFree(d_outputPos));
//	checkCudaErrors(cudaFree(d_outputVals));
//}

