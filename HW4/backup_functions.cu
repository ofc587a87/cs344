/*



__global__ void calcHistogramGlobal(unsigned int const d_dataVals[], unsigned int d_histogram[], const unsigned int iteration, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if(myId<numElems)
	{
		bool isOne=isBitByRight(d_dataVals[myId], iteration);
		atomicAdd(&(d_histogram[isOne?1:0]), 1);
	}
	__syncthreads();
}

__global__ void calcHistogramGlobalBatch(unsigned int const d_dataVals[], unsigned int d_histogram[], const unsigned int iteration, const size_t numElems, const int batchSize)
{
	int tid=threadIdx.x;
	unsigned int l_histogram_ZERO=0;
	unsigned int l_histogram_ONE=0;

	for(int i=0;i<batchSize;i++)
	{
		int myId=(tid + (i * blockDim.x));
		if(myId<numElems)
		{
			if(isBitByRight(d_dataVals[myId], iteration))
				l_histogram_ONE+=1;
			else
				l_histogram_ZERO+=1;
		}
	}

	atomicAdd(&(d_histogram[0]), l_histogram_ZERO);
	atomicAdd(&(d_histogram[1]), l_histogram_ONE);
}






__global__ void setIntermediate(unsigned int const *d_dataVals, unsigned int* d_intermediate,
		const unsigned int iteration, const size_t numElems, bool isStep1)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if(myId<numElems)
	{
		bool isOne=isBitByRight(d_dataVals[myId], iteration);
		d_intermediate[myId] = (isOne?(isStep1?0:1):(isStep1?1:0));
	}
	__syncthreads();
}

__global__ void compactGlobal(unsigned int const *d_dataVals,
        unsigned int const *d_histogram,
        const size_t numElems, const unsigned int iteration,
        unsigned int* d_intermediate, unsigned int *d_scatterAddress, bool step1, int step)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

	if(myId<numElems)
	{
		bool isOne=isBitByRight(d_dataVals[myId], iteration);

		//Step 1: evaluate value 0, step 2: evaluate value 1
		//scan using predicate to compute scatter address
		//Step hillis / Steele
		if (myId >=step)
		{
			d_intermediate[myId] += d_intermediate[myId - step];
		}
	}

}

__global__ void compactGlobalBatch(unsigned int const *d_dataVals,
        unsigned int const *d_histogram,
        const size_t numElems, const unsigned int iteration,
        unsigned int* d_intermediate, unsigned int *d_scatterAddress, int batchSize)
{
	int tid=threadIdx.x;

	for(int i=0;i<batchSize;i++)
	{
		int myId=(tid + (i * blockDim.x));

		if(myId<numElems)
	{
		bool isOne=isBitByRight(d_dataVals[myId], iteration);

		//Step 1: evaluate value 0, step 2: evaluate value 1
		//scan using predicate to compute scatter address
		//Step hillis / Steele
		if (myId >=step)
		{
			d_intermediate[myId] += d_intermediate[myId - step];
		}
	}

}

__global__ void copyCompactScatter(unsigned int const *d_dataVals, const unsigned int iteration,
        unsigned int* d_intermediate, unsigned int *d_scatterAddress, bool step1)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	bool isOne=isBitByRight(d_dataVals[myId], iteration);
	//copy to final scatter address
	if((isOne && step1) || (!isOne && !step1))
		d_scatterAddress[myId]=d_intermediate[myId]-1;
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

__global__ void scatterGlobal(unsigned int* const d_inputVals, unsigned int* d_outputVals,
		unsigned int* d_scatterAddress, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

	int maxStepCompact = 1;
	// Double powof2 until >= val
	while( maxStepCompact < numElems ) maxStepCompact <<= 1;

	if(myId<numElems)
	{
		unsigned int address=d_scatterAddress[myId];
//		if(address < numElems)
//			d_outputVals[address]=d_inputVals[myId];
		d_outputVals[myId]=address < numElems?myId:(10000000 + address);
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

//__global__ void sortPixelsGlobal(unsigned int* const d_inputVals, unsigned int* const d_inputPos,
//        unsigned int* const d_outputVals, unsigned int* const d_outputPos,
//        const size_t numElems,
//        unsigned int* d_histogram, unsigned int* d_intermediate, unsigned int* d_scatterAddress, const unsigned int iteration)
//{
//
//	int myId = threadIdx.x + blockDim.x * blockIdx.x;
//
//	//only use the neccesary threads
//	if(myId>=numElems) return;
//	d_outputVals[myId] = d_inputVals[myId];
//	d_outputPos[myId] = d_inputPos[myId];
//
//	//Step1  outside, calc histogram
//
//	//Step 2: Compact t calc scatter address
//	compactGlobal(d_outputVals, d_histogram[0], numElems, iteration, d_intermediate, d_scatterAddress);
//	__syncthreads();
//
//	//step 3: scatter
//	scatterGlobal(d_outputVals, d_intermediate, d_scatterAddress, numElems);
//	__syncthreads();
//	d_outputVals[myId]=d_intermediate[myId];
//	__syncthreads();
//
//	scatterGlobal(d_outputPos, d_intermediate, d_scatterAddress, numElems);
//	__syncthreads();
//	d_outputPos[myId]=d_intermediate[myId];
//	__syncthreads();
//
//}

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


void doCompact( unsigned int* const d_outputVals, unsigned int *d_intermediate,
		unsigned int *d_histogram, unsigned int *d_scatterAddress,
		const size_t numElems, int iteration, unsigned int NUM_BLOCKS, unsigned int NUM_THREADS, bool step1)
{


	cudaMemset(d_intermediate, 0, sizeof(unsigned int) * numElems);
	cudaMemset(d_scatterAddress, 0, sizeof(unsigned int) * numElems);
	setIntermediate<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_intermediate, iteration, numElems, step1);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	for (int step = 1; step<maxStepCompact;step <<= 1)
	{
		compactGlobal<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, d_histogram, numElems, iteration, d_intermediate, d_scatterAddress, step1, step);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	copyCompactScatter<<<NUM_BLOCKS, NUM_THREADS>>>(d_outputVals, iteration, d_intermediate, d_scatterAddress, step1);

}
*/
