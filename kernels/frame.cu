#include <cuda.h>
#include <iostream>
#include "frame.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


int
Frame::load_from_file(std::string filename)
{
	this->filename = filename;
	// request the image as grayscale
	this->data = stbi_load(filename.c_str(), &this->x, &this->y, &this->n, 1);
	this->memsize = x * y * sizeof(unsigned char);
	if (!data) {
		// error happened, handle accordingly!
		std::cerr << "EE: File '" << filename << "' corrupt. Could not read data." << std::endl;
		return 1;
	}

	// std::cout << "II: Frame (" << x << ", " << y << ")" << std::endl;

	cudaError_t errcode = cudaMalloc((void**)&dev_data, memsize);
	if (errcode != cudaSuccess) {
		std::cerr << "EE: Could not allocate CUDA Device memory for Frame" << std::endl;
		return 1;
	}
	return 0;
}


void
Frame::toGPU()
{
	cudaError_t errcode = cudaMemcpy(dev_data, data, memsize, cudaMemcpyHostToDevice);
	if (errcode != cudaSuccess) {
		std::cerr << "EE: Error while transfering frame data to CUDA Device" << std::endl;
	}
}


Frame::
~Frame() {
	if (dev_data) cudaFree(dev_data);
	if (data) stbi_image_free(data);
}
