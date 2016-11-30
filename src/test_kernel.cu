/*
  ALLOCATION = SHARED
  DENSE      = SPARSE
  MSCT       = 1
  unified kernel
*/


__device__ void
stencilX (int *temp_idx_shared, float *temp_beta_shared, float *E, int iter, int swap_mod, int mod)
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB

  curandState seed0 = pgpuseed[TperB * blockIdx.x + bidx];	// curand sequence


  // signed 16 bit integer: -32K ~ 32K, never overflows
  // sizeof (shot) * 24 * 512 = 24 KB
  __shared__ short E_shared[NBETA_PER_WORD][TperB];



  /// temperature scratchpad
  __shared__ PROB_DATATYPE temp_prob_shared[NBETA_PER_WORD][NPROB_MAX];
  gpu_init_temp (temp_prob_shared, bidx);


  /// lattice scratchpad
  // sizeof(u_int32_t) * 16 * 16 * 16 = 16 KB
  __shared__ MSC_DATATYPE l[SZ][SZ][SZ];

  // index for read/write glocal memory
  const int xx = (SZ_HF * (threadIdx.y & 1)) + threadIdx.x;
  const int yy = (SZ_HF * (threadIdx.z & 1)) + (threadIdx.y >> 1);

  // index for reading scratchpad
  const int y = threadIdx.y;
  const int ya = (y + SZ - 1) % SZ;
  const int yb = (y + 1) % SZ;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // initilize temperature scratchpad
    gpu_compute_temp (temp_prob_shared, temp_beta_shared, bidx, word);

    // import lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += (BDz0 >> 1)) {
      int zz = z_offset + (threadIdx.z >> 1);
      l[zz][yy][xx] = plattice[lattice_offset + SZ * SZ * z_offset + bidx];
    }

    // reset partial status
    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b][bidx] = 0;

    __syncthreads ();



    for (int i = 0; i < iter; i++) {
      int mybool = (mod == 1) && (i % ITER_SWAP_KERNFUNC == 0);

      // two phases update
      for (int run = 0; run < 2; run++) {
	int x0 = (threadIdx.z & 1) ^ (threadIdx.y & 1) ^ run;	// initial x
	int x = (threadIdx.x << 1) + x0;
	int xa = (x + SZ - 1) % SZ;
	int xb = (x + 1) % SZ;
	//int xa = (x + SZ - 1) & (SZ - 1);
	//int xb = (x + 1) & (SZ - 1);

	// data reuse among z ???
	for (int z_offset = 0; z_offset < SZ; z_offset += BDz0) {
	  int z = z_offset + threadIdx.z;
	  int za = (z + SZ - 1) % SZ;
	  int zb = (z + 1) % SZ;
	  //int za = (z + SZ - 1) & (SZ - 1);
	  //int zb = (z + 1) & (SZ - 1);

	  MSC_DATATYPE c = l[z][y][x];	// center
	  MSC_DATATYPE n0 = l[z][y][xa];	// left
	  MSC_DATATYPE n1 = l[z][y][xb];	// right
	  MSC_DATATYPE n2 = l[z][ya][x];	// up
	  MSC_DATATYPE n3 = l[z][yb][x];	// down
	  MSC_DATATYPE n4 = l[za][y][x];	// front
	  MSC_DATATYPE n5 = l[zb][y][x];	// back

	  n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
	  n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
	  n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
	  n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
	  n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
	  n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

	  // process a lattice element
	  MSC_DATATYPE flip = 0;



	  for (int b = 0; b < NBETA_PER_WORD; b++) {
	    int energy =	// range: [0,6]
	      (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
	      (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
	    int spin = c >> b & 1;
	    energy = energy * 2 - 6;
	    spin = spin * 2 - 1;
	    PROB_DATATYPE val = expf (2 * energy * temp_beta_shared[b]);	    
	    PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	    int flippp = myrand < val;
	    flip |= (flippp << b);	// myrand < val ? 1 : 0;
	    if (mybool)
	      E_shared[b][bidx] += (energy + spin) * (flippp * 2 - 1);
	  }


	  l[z][y][x] = c ^ flip;
	}			// z_offset

	__syncthreads ();
      }				// run

      if (mybool) {
	gpu_reduction (E, E_shared, bidx, word);
	__syncthreads ();

	gpu_shuffle (temp_idx_shared, temp_beta_shared, E, bidx, mod);
	__syncthreads ();
      }

    }				// i


    // export lattice scratchpad
    for (int z_offset = 0; z_offset < SZ; z_offset += (BDz0 >> 1)) {
      int zz = z_offset + (threadIdx.z >> 1);
      plattice[lattice_offset + SZ * SZ * z_offset + bidx] = l[zz][yy][xx];
    }

    __syncthreads ();
  }				// word


  // copy seed back
  pgpuseed[TperB * blockIdx.x + bidx] = seed0;
}






// rearrange the spins so that they matches the temperature order
// least significant bit - lowest temperature
// higher order bits - higher temperature
// stored in compact format, regardless of MSCT

__global__ void
kernel_rearrange ()
{
  const int bidx = threadIdx.x;

  // temperature scratchpad
  __shared__ int __align__ (32) temp_idx_shared[NBETA_PER_WORD];

  // initilize lattice1
  for (int offset = 0; offset < SZ_CUBE; offset += TperB)
    plattice1[SZ_CUBE * blockIdx.x + offset + bidx] = 0;
  __syncthreads ();


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = (SZ_CUBE * NWORD * blockIdx.x) + (SZ_CUBE * word);

    if (bidx < NBETA_PER_WORD)
      temp_idx_shared[bidx] =
	ptemp[NBETA_MAX * blockIdx.x + NBETA_PER_WORD * word + bidx].idx;
    __syncthreads ();

    for (int offset = 0; offset < SZ_CUBE; offset += TperB) {
      MSC_DATATYPE oldword = plattice[lattice_offset + offset + bidx];
      MSC_DATATYPE newword = 0;
      for (int b = 0; b < NBETA_PER_WORD; b++) {
	MSC_DATATYPE tmp = oldword >> (MSCT * b) & 1;
	tmp <<= temp_idx_shared[b];
	newword |= tmp;
      }
      plattice1[SZ_CUBE * blockIdx.x + offset + bidx] |= newword;
    }

    __syncthreads ();
  }

}







// initilize seeds for curand
// CURAND_Library.pdf, pp21
__global__ void
kernel_init_seed ()
{
  const int gidx = TperB * blockIdx.x + threadIdx.x;
  curand_init (seed, gidx, 0, &pgpuseed[gidx]);

  // seed, subsequence, offset, gpuseed
  // skipahead(100000, &gpuseed[gidx]);
}







__global__ void
kernel_unified (int rec, int mod)
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB

  /// temperature
  // (4 * 32) * 2 = 256 B
  __shared__ int __align__ (32) temp_idx_shared[NBETA];
  __shared__ float __align__ (32) temp_beta_shared[NBETA];

  /// lattice energy
  // sizeof (float) * 32 = 128 B
  __shared__ float __align__ (32) E[NBETA];

  // load temperature
  if (bidx < NBETA) {
    temp_idx_shared[bidx] = ptemp[NBETA_MAX * blockIdx.x + bidx].idx;
    temp_beta_shared[bidx] = ptemp[NBETA_MAX * blockIdx.x + bidx].beta;
  }


  if (mod == 0)
    for (int i = 0; i < ITER_WARMUP_KERN; i += ITER_WARMUP_KERNFUNC) {
      stencilX (temp_idx_shared, temp_beta_shared, E, ITER_WARMUP_KERNFUNC, 0, 0);
    }
  else
    for (int i = 0; i < ITER_SWAP_KERN; i += ITER_SWAP_KERNFUNC) {
      int swap_mod = (i / ITER_SWAP_KERNFUNC) & 1;
      stencilX (temp_idx_shared, temp_beta_shared, E, ITER_SWAP_KERNFUNC, swap_mod, 1);
    }



  // store temperature
  if (bidx < NBETA) {
    ptemp[NBETA_MAX * blockIdx.x + bidx].idx = temp_idx_shared[bidx];
    ptemp[NBETA_MAX * blockIdx.x + bidx].beta = temp_beta_shared[bidx];
  }
  __syncthreads ();

  // store energy status
  //  if (bidx < NBETA)
  //  pst[rec].e[blockIdx.x][temp_idx_shared[bidx]] = E[bidx];
  //__syncthreads ();
}




__global__ void
kernel_compute_q (int rec)
{
  const int bidx = threadIdx.x;

  // sizeof(u_int32_t) * 16 * 16 * 16 = 16 KB 
  __shared__ MSC_DATATYPE l1[SZ_CUBE];

  const int lattice_offset0 = SZ_CUBE * (blockIdx.x << 1);
  const int lattice_offset1 = lattice_offset0 + SZ_CUBE;

  for (int offset = 0; offset < SZ_CUBE; offset += TperB) {
    l1[offset + bidx] =		// xord_word 
      plattice1[lattice_offset0 + offset + bidx] ^
      plattice1[lattice_offset1 + offset + bidx];
  }

  // is double an overkill?
  if (bidx < NBETA) {
    float q0 = 0.0f;
    double qk_real = 0.0;
    double qk_imag = 0.0;
    double qk2_real = 0.0;
    double qk2_imag = 0.0;
    double angel1, angel2;

    MSC_DATATYPE xor_word;
    int xor_bit;

    for (int i = 0; i < SZ_CUBE; i++) {
      xor_word = l1[i];
      xor_bit = (xor_word >> bidx) & 0x1;
      xor_bit = 1 - (xor_bit << 1);	// parallel: +1, reverse: -1

      // 2 * pi / L * x_i
      angel1 = (double) (i % SZ) * 2 * PI / SZ;
      // 2 * pi / L * (x_i + y_i)
      angel2 = (double) (i % SZ + (i / SZ) % SZ) * 2 * PI / SZ;

      q0 += xor_bit;
      qk_real += xor_bit * cos (angel1);
      qk_imag += xor_bit * sin (angel1);
      qk2_real += xor_bit * cos (angel2);
      qk2_imag += xor_bit * sin (angel2);
    }

    // save measurements in "st"
    pst[rec].q[blockIdx.x][bidx] = q0;
    pst[rec].qk[blockIdx.x][bidx] =
      (float) sqrt (qk_real * qk_real + qk_imag * qk_imag);
    pst[rec].qk2[blockIdx.x][bidx] =
      (float) sqrt (qk2_real * qk2_real + qk2_imag * qk2_imag);
  }
}
