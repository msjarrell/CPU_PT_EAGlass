#include "COPYING"


/*
  ALLOCATION = INTEGRATED
  DENSE      = COMPACT
  MSCT       = 4
*/


__device__ void
stencil (float *temp_beta_shared, int iter)
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB
  curandState seed0 = pgpuseed[TperB * blockIdx.x + bidx];

  /// temperature scratchpad
  __shared__ float temp_prob_shared[NBETA_PER_WORD][NPROB_MAX];
  gpu_init_temp (temp_prob_shared, bidx);

  /// lattice scratchpad
  // sizeof(u_int32_t) * 16 * 16 * 8 = 8 KB
  __shared__ MSC_DATATYPE l[SZ][SZ][SZ_HF];

  // index for reading scratchpad
  const int y = threadIdx.y;
  const int ya = (y + SZ - 1) % SZ;
  const int yb = (y + 1) % SZ;
  const int x = threadIdx.x;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE_HF * NWORD * blockIdx.x + SZ_CUBE_HF * word;

    // initilize temperature scratchpad
    gpu_compute_temp (temp_prob_shared, temp_beta_shared, bidx, word);

    // import lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      l[z][y][x] = plattice[lattice_offset + (SZ_HF * SZ * z_offset) + bidx];
    }
    __syncthreads ();



    for (int i = 0; i < iter; i++) {

      // two phases update
      for (int run = 0; run < 2; run++) {
	int x0 = (threadIdx.z & 1) ^ (threadIdx.y & 1) ^ run;	// initial x
	int xa = (x + SZ_HF - !x0) % SZ_HF;
	int xb = (x + x0) % SZ_HF;

	// data reuse among z ???
	for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
	  int z = z_offset + threadIdx.z;
	  int za = (z + SZ - 1) % SZ;
	  int zb = (z + 1) % SZ;
	  //int za = (z + SZ - 1) & (SZ - 1);
	  //int zb = (z + 1) & (SZ - 1);

	  MSC_DATATYPE n0 = l[z][y][xa] >> !run;	// left
	  MSC_DATATYPE n1 = l[z][y][xb] >> !run;	// right
	  MSC_DATATYPE n2 = l[z][ya][x] >> !run;	// up
	  MSC_DATATYPE n3 = l[z][yb][x] >> !run;	// down
	  MSC_DATATYPE n4 = l[za][y][x] >> !run;	// front
	  MSC_DATATYPE n5 = l[zb][y][x] >> !run;	// back



#if MSC_FORMAT == 0
	  MSC_DATATYPE c = l[z][y][x];       // center

	  n0 = MASK_A * ((c >> (SHIFT_J0 + 6 * run)) & 1) ^ n0 ^ (c >> run);
	  n1 = MASK_A * ((c >> (SHIFT_J1 + 6 * run)) & 1) ^ n1 ^ (c >> run);
	  n2 = MASK_A * ((c >> (SHIFT_J2 + 6 * run)) & 1) ^ n2 ^ (c >> run);
	  n3 = MASK_A * ((c >> (SHIFT_J3 + 6 * run)) & 1) ^ n3 ^ (c >> run);
	  n4 = MASK_A * ((c >> (SHIFT_J4 + 6 * run)) & 1) ^ n4 ^ (c >> run);
	  n5 = MASK_A * ((c >> (SHIFT_J5 + 6 * run)) & 1) ^ n5 ^ (c >> run);

	  // for profiling purpose
	  //float val = 0.7;
	  //float myrand = curand_uniform (&seed0);
	  //float myrand = 0.4;
	  //c = c ^ n0 ^ n1 ^ n2 ^ n3 ^ n4 ^ n5;

	  ///*
	  for (int s = 0; s < 4; s+=2) {
	    MSC_DATATYPE e =
	      ((n0 >> s) & MASK_S) +
	      ((n1 >> s) & MASK_S) +
	      ((n2 >> s) & MASK_S) +
	      ((n3 >> s) & MASK_S) +
	      ((n4 >> s) & MASK_S) +
	      ((n5 >> s) & MASK_S);
	    e = (e << 1) + ((c >> s) & MASK_S);
	    MSC_DATATYPE flip = 0;
	    //#pragma unroll
	    for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	      PROB_DATATYPE val = temp_prob_shared[shift + (s >> 2)][(e >> shift) & MASK_E];
	      //PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	      PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	      flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	    }
	    c ^= (flip << s);
	  }

	  l[z][y][x] = c;
	  //*/
#endif


#if MSC_FORMAT == 1
	  MSC_DATATYPE c = l[z][y][x] >> run;       // center

	  n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
	  n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
	  n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
	  n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
	  n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
	  n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

	  // for profiling purpose
	  //float val = 0.7;
	  //float myrand = curand_uniform (&seed0);
	  //float myrand = 0.4;
	  //c = c ^ n0 ^ n1 ^ n2 ^ n3 ^ n4 ^ n5;

	  ///*
	  MSC_DATATYPE e =
	    (n0 & MASK_S) +
	    (n1 & MASK_S) +
	    (n2 & MASK_S) +
	    (n3 & MASK_S) +
	    (n4 & MASK_S) +
	    (n5 & MASK_S);
	  e = (e << 1) + (c & MASK_S);

	  MSC_DATATYPE flip = 0;
	  //#pragma unroll
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    //PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }

	  l[z][y][x] ^= (flip << run);
	  //*/
#endif


	}			// z_offset


	__syncthreads ();
      }				// run
    }				// i



    // export lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      plattice[lattice_offset + (SZ_HF * SZ * z_offset) + bidx] = l[z][y][x];
    }
    __syncthreads ();

  }				// word


  // copy seed back
  pgpuseed[TperB * blockIdx.x + bidx] = seed0;
}





__device__ void
stencil_swap (int *temp_idx_shared, float *temp_beta_shared, float *E, int mod)
{
}





// rearrange the spins so that they matches the temperature order
// least significant bit - lowest temperature
// higher order bits - higher temperature
// stored in compact format, regardless of MSCT

__global__ void
kernel_rearrange ()
{

}

