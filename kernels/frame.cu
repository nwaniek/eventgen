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

	cudaMalloc((void**)&dev_data, memsize);
	return 0;
}

void
Frame::toGPU()
{
	cudaMemcpy(dev_data, data, memsize, cudaMemcpyHostToDevice);
}

Frame::
~Frame() {
	if (dev_data) cudaFree(dev_data);
	if (data) stbi_image_free(data);
}
