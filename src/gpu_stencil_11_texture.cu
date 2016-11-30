#include "COPYING"


/*
  ALLOCATION = SEPARATED
  DENSE      = COMPACT
  MSCT       = 4
*/


/*
texture<float,1,cudaReadModeElementType> mytexture
tex1Dfetch(mytexture, ide+2*DIM)
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
  __shared__ MSC_DATATYPE l0[SZ][SZ][SZ_HF];
  __shared__ MSC_DATATYPE l1[SZ][SZ][SZ_HF];

  // index for reading scratchpad
  const int y = threadIdx.y;
  const int ya = (y + SZ - 1) % SZ;
  const int yb = (y + 1) % SZ;
  const int x = threadIdx.x;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // initilize temperature scratchpad
    gpu_compute_temp (temp_prob_shared, temp_beta_shared, bidx, word);

    // import lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      int index = lattice_offset + (SZ_HF * SZ * z_offset) + bidx;
      l0[z][y][x] = plattice[index];
      l1[z][y][x] = plattice[index + SZ_CUBE_HF];
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

	  MSC_DATATYPE c, n0, n1, n2, n3, n4, n5;

	  if (run == 0) {
	    c = l0[z][y][x];	// center
	    n0 = l1[z][y][xa];	// left
	    n1 = l1[z][y][xb];	// right
	    n2 = l1[z][ya][x];	// up
	    n3 = l1[z][yb][x];	// down
	    n4 = l1[za][y][x];	// front
	    n5 = l1[zb][y][x];	// back
	  }
	  else {
	    c = l1[z][y][x];	// center
	    n0 = l0[z][y][xa];	// left
	    n1 = l0[z][y][xb];	// right
	    n2 = l0[z][ya][x];	// up
	    n3 = l0[z][yb][x];	// down
	    n4 = l0[za][y][x];	// front
	    n5 = l0[zb][y][x];	// back
	  }

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
	  for (int s = 0; s < NBETA_PER_SEG; s++) {
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
	      PROB_DATATYPE val = temp_prob_shared[shift + s][(e >> shift) & MASK_E];
	      //PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	      PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	      flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	    }
	    c ^= (flip << s);
	  }
	  //*/

	  if (run == 0)
	    l0[z][y][x] = c;
	  else
	    l1[z][y][x] = c;
	}			// z_offset


	__syncthreads ();
      }				// run
    }				// i



    // export lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      int index = lattice_offset + (SZ_HF * SZ * z_offset) + bidx;
      plattice[index] = l0[z][y][x];
      plattice[index + SZ_CUBE_HF] = l0[z][y][x];
    }
    __syncthreads ();

  }				// word


  // copy seed back
  pgpuseed[TperB * blockIdx.x + bidx] = seed0;
}





__device__ void
stencil_swap (int *temp_idx_shared, float *temp_beta_shared, float *E, int mod)
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB

  /// E scratchpads
  // does "short" datatype degrade performance?

  // signed 16 bit integer: -32K ~ 32K, never overflows
  // sizeof (shot) * 24 * 512 = 24 KB
  __shared__ short E_shared[NBETA_PER_WORD][TperB];
  // sizeof (float) * 32 = 128 B
  __shared__ float __align__ (32) Eh[NBETA];


  /// lattice scratchpad
  // sizeof(u_int32_t) * 16 * 16 * 8 = 8 KB
  __shared__ MSC_DATATYPE l0[SZ][SZ][SZ_HF];
  __shared__ MSC_DATATYPE l1[SZ][SZ][SZ_HF];

  // index for reading scratchpad
  const int y = threadIdx.y;
  const int ya = (y + SZ - 1) % SZ;
  const int yb = (y + 1) % SZ;
  const int x = threadIdx.x;
  const int xa = (x + SZ_HF - 1) % SZ_HF;
  const int xb = (x + 1) % SZ_HF;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // import lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      int index = lattice_offset + (SZ_HF * SZ * z_offset) + bidx;
      l0[z][y][x] = plattice[index];
      l1[z][y][x] = plattice[index + SZ_CUBE_HF];
    }

    // reset partial status
    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b][bidx] = 0;

    __syncthreads ();


    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      int za = (z + SZ - 1) % SZ;
      int zb = (z + 1) % SZ;
      //int za = (z + SZ - 1) & (SZ - 1);
      //int zb = (z + 1) & (SZ - 1);

      MSC_DATATYPE c = l0[z][y][x];	// center
      MSC_DATATYPE n0 = l1[z][y][xa];	// left
      MSC_DATATYPE n1 = l1[z][y][xb];	// right
      MSC_DATATYPE n2 = l1[z][ya][x];	// up
      MSC_DATATYPE n3 = l1[z][yb][x];	// down
      MSC_DATATYPE n4 = l1[za][y][x];	// front
      MSC_DATATYPE n5 = l1[zb][y][x];	// back

      n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
      n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
      n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
      n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
      n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
      n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

      for (int s = 0; s < NBETA_PER_SEG; s++) {
	MSC_DATATYPE e =
	  ((n0 >> s) & MASK_S) +
	  ((n1 >> s) & MASK_S) +
	  ((n2 >> s) & MASK_S) +
	  ((n3 >> s) & MASK_S) +
	  ((n4 >> s) & MASK_S) +
	  ((n5 >> s) & MASK_S);
	//#pragma unroll
	for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	  E_shared[shift + s][bidx] += (e >> shift) & MASK_E;	// range: [0,6]
	}
      }

    }				// z_offset

    //__syncthreads ();
    gpu_reduction (E, E_shared, bidx, word);
    __syncthreads ();




    /// energy contribute by external field

    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b][bidx] = 0;
    
    for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
      int z = z_offset + threadIdx.z;
      MSC_DATATYPE c0 = l0[z][y][x];
      MSC_DATATYPE c1 = l1[z][y][x];

      for (int s = 0; s < NBETA_PER_SEG; s++) {
	for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	  int ss = shift + s;
	  E_shared[ss][bidx] += ((c0 >> ss) & 1) + ((c1 >> ss) & 1);
	}
      }
    }

    gpu_reduction (Eh, E_shared, bidx, word);
    __syncthreads ();

  }				// word;



  // convert E from [0,6] to [-6,6], e = e * 2 - 6
  // E = sum_TperB sum_ZITER (e * 2 - 6)
  //   = 2 * sum_ZITER_TperG e - 6 * ZITER * TperB

  // conver Eh from [0,1] to [-1,1], e = e * 2 - 1
  // Eh = 2 * sum_ZITER_TperG e - SZ_CUBE


  // donot need to substrasct the constant
  if (bidx < NBETA) {
    E[bidx] = E[bidx] * 2 - 6 * (SZ / BDz0) * TperB;
    Eh[bidx] = Eh[bidx] * 2 - SZ_CUBE;
    E[bidx] = E[bidx] + Eh[bidx] * H;
  }
  __syncthreads ();


  gpu_shuffle (temp_idx_shared, temp_beta_shared, E, bidx, mod);
  __syncthreads ();
}








// rearrange the spins so that they matches the temperature order
// least significant bit - lowest temperature
// higher order bits - higher temperature
// stored in compact format, regardless of MSCT

__global__ void
kernel_rearrange ()
{

}



