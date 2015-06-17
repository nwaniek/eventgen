#include <cuda.h>
#include <iostream>
#include <utility>
#include "dvs.h"
#include "frame.h"


/*
 * struct EventBuffer - store a certain amount of events in the form of a buffer
 */
struct EventBuffer {
	int counter;
	dvs_event_t events[];
};



__global__
void dvs_sim(
	int width, int height,
	unsigned char *left, unsigned char *right,
	int thresh, EventBuffer *buffer, uint64_t t)
{
	// as event generation is sparse, we can simply use atomicAdd here
	// without too much time penalty

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) return;
	int idx = y * width + x;

	// int diff = (int)left[pxl] - (int)right[pxl];
	int diff = (int)left[idx] - (int)right[idx];

	// on-event
	if (diff > thresh)
		buffer->events[atomicAdd(&buffer->counter, 1)] = {1u, (uint16_t)x, (uint16_t)y, t};
	// off-event
	else if (diff < -thresh)
		buffer->events[atomicAdd(&buffer->counter, 1)] = {0u, (uint16_t)x, (uint16_t)y, t};
}


void
print_cuda_info()
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	std::cout << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl
		  << "maxThreadsDim:      [" <<
			props.maxThreadsDim[0] << ", " <<
			props.maxThreadsDim[1] << ", " <<
			props.maxThreadsDim[2] << "]" << std::endl
		  << "maxGridSize:        [" <<
			props.maxGridSize[0] << ", " <<
			props.maxGridSize[1] << ", " <<
			props.maxGridSize[2] << "]" << std::endl
		  ;
}


std::vector<dvs_event_t>
process_files(config_t &config, std::vector<std::string> &files)
{
	Frame *left = new Frame();
	left->load_from_file(files[0]);
	left->toGPU();
	int N = left->x * left->y;

	// allocate memory to store the events both on host and device
	size_t bufsize = sizeof(int) + N * sizeof(dvs_event_t);
	EventBuffer *buf_a = (EventBuffer*)malloc(bufsize);
	EventBuffer *buf_b = (EventBuffer*)malloc(bufsize);
	memset(buf_a, 0, bufsize);
	memset(buf_b, 0, bufsize);

	EventBuffer *dev_buf_a;
	EventBuffer *dev_buf_b;
	cudaMalloc((void**)&dev_buf_a, bufsize);
	cudaMalloc((void**)&dev_buf_b, bufsize);

	// storage for the result
	std::vector<dvs_event_t> result;

	int64_t t = config.start_t;
	for (size_t i = 1; i < files.size(); i++) {
		Frame *right = new Frame();
		right->load_from_file(files[i]);

		// synchronization point: load data to/from GPU
		right->toGPU();
		cudaMemcpy(dev_buf_a, buf_a, bufsize, cudaMemcpyHostToDevice);
		cudaMemcpy(buf_b, dev_buf_b, bufsize, cudaMemcpyDeviceToHost);

		// call the CUDA kernel
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks(left->x / threadsPerBlock.x, left->y / threadsPerBlock.y);
		dvs_sim<<<numBlocks, threadsPerBlock>>>(left->x, left->y, left->dev_data, right->dev_data,
				config.thresh, dev_buf_a, t);

		// copy the events to the result vector
		for (int i = 0; i < buf_b->counter; i++)
			result.push_back(buf_b->events[i]);

		// reset the event buffer counter
		buf_b->counter = 0;

		// wait for the device to finish
		cudaDeviceSynchronize();

		// swap pointers
		std::swap(dev_buf_a, dev_buf_b);
		std::swap(buf_a, buf_b);
		std::swap(left, right);

		t += config.delta_t;
	}

	free(buf_b);
	free(buf_a);

	return std::move(result);
}
