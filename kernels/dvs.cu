#include <cuda.h>
#include <iostream>
#include "dvs.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__global__
void to_grayscale(unsigned char *img, unsigned char *result)
{
	if (threadIdx.x > 127 || threadIdx.y > 127) return;

	int i = threadIdx.x * 127 + threadIdx.y;
	float r, g, b, gray;
	r = img[i + 0];
	g = img[i + 1];
	b = img[i + 2];

	gray = 0.299f*r + 0.587*g + 0.114f*b;


}


__global__
void dvs_sim(float *x, int i)
{

}

// invoke the cuda kernel
// dvs_sim<<<127, 127>>>(NULL, 0);


int
process_files(config_t &config, std::vector<std::string> &files)
{
	for (auto f: files) {
		int x, y, n;
		unsigned char *data = stbi_load(f.c_str(), &x, &y, &n, 0);
		stbi_image_free(data);
	}

	// iterate over pairs of


	// pass all the files to the CUDA kernel

	/*
	const char *fname = "/home/rochus/data/FlorianScherer-OpticFlow-ImageSequences/Frames/0000.png";

	int x, y, n;
	unsigned char *data = stbi_load(fname, &x, &y, &n, 0);
	std::cout << x << ", " << y << ", " << n << std::endl;

	stbi_image_free(data);

	// call_wrapper();
	return EXIT_SUCCESS;
	*/



	return EXIT_SUCCESS;
}
