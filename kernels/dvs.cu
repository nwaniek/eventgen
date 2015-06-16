#include <cuda.h>
#include <iostream>
#include <utility>
#include "dvs.h"
#include "frame.h"

__global__
void dvs_sim(float *x, int i)
{

}

// invoke the cuda kernel
// dvs_sim<<<127, 127>>>(NULL, 0);




int
process_files(config_t &config, std::vector<std::string> &files)
{
	Frame *left = new Frame();
	left->load_from_file(files[0]);

	for (size_t i = 1; i < files.size(); i++) {
		Frame *right = new Frame();
		right->load_from_file(files[i]);




		std::swap(left, right);
	}
	return EXIT_SUCCESS;
}
