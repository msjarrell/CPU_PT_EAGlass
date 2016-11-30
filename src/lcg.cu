#include "lcg.cuh"


// Linear congruential generator

// passing pointer paramter
// Will this force allocation in memory?
// expect register variables.


__forceinline__
__device__ void
gpu_lcg (LCG_DATATYPE* seed)
{
  *seed = LCG_A * (*seed % LCG_Q) - LCG_R * (*seed / LCG_Q);
  *seed += LCG_M * (*seed < 0);
}

