#include "misc.h"
#include <cuda.h>


void
initCuda() {
	// canonical way to establish a CUDA context...
	cudaFree(0);
}
