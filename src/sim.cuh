#include "COPYING"


#ifndef SIM_CUH
#define SIM_CUH


// gpu_func.cc
__device__ void gpu_init_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], const int bidx);
__device__ void gpu_compute_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], float *temp_beta_shared, const int bidx, int word);
__device__ void gpu_shuffle (int *temp_idx_shared, float *temp_beta_shared, float *E, const int bidx, int mod);
__device__ void gpu_reduction (float *a, short a_shared[NBETA_PER_WORD][TperB], const int bidx, int word);


// gpu_stencil.cu
__device__ void stencil (float *temp_beta_shared, int iter);
__device__ void stencil_swap (int *temp_idx_shared, float *temp_beta_shared, float *E, int mod);
__global__ void kernel_rearrange ();


// gpu_kernel.cu
__global__ void kernel_init_seed ();
__global__ void kernel_unified (int rec, int mod);
__global__ void kernel_warmup ();
__global__ void kernel_swap (int rec);
__global__ void kernel_compute_q (int rec);


#endif /* SIM_CUH */
