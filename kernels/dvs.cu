#include <cuda.h>

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


void call_wrapper() {

	// invoke the cuda kernel
	dvs_sim<<<127, 127>>>(NULL, 0);

}
