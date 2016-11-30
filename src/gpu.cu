#include "COPYING"


#include <stdlib.h>
#include <stdio.h>
//#include <stdint.h>
//#include <math.h>
//#include <time.h>
//#include <ctype.h>
//#include <sys/stat.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "sim.h"
#include "sim.cuh"
#include "lcg.cuh"
#include "Random123/philox.h"
#include "mylimits.h"


__constant__ int seed;
__constant__ curandState *pgpuseed;
__constant__ LCG_DATATYPE *pgpuseed1;
__constant__ MSC_DATATYPE *plattice;
__constant__ MSC_DATATYPE *plattice1;
__constant__ Temp *ptemp;
__constant__ St *pst;
__constant__ int *pcnt;



// How to split function into seperate source file?
// NVIDIA SDK does not link multiple file.
// "#include *.c" is a quick dirty solution

#include "lcg.cu"
#include "gpu_func.cu"


#if 1
#if ALLOCATION == SHARED && DENSE == SPARSE
#include "gpu_stencil_00.cu"
#endif

#if ALLOCATION == SHARED && DENSE == COMPACT
#include "gpu_stencil_01.cu"
#endif

#if ALLOCATION == SEPARATED && DENSE == COMPACT
#include "gpu_stencil_11.cu"
#endif

#if ALLOCATION == INTEGRATED && DENSE == COMPACT
#include "gpu_stencil_21.cu"
#endif

#if ALLOCATION == INTEGRATED2 && DENSE == COMPACT
#include "gpu_stencil_31.cu"
#endif
#endif


//#include "test_rand.cu"

#include "gpu_kernel.cu"

//#include "test_kernel.cu"

