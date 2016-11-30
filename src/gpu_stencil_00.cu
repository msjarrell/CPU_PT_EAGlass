#include "COPYING"


/*
  ALLOCATION = SHARED
  DENSE      = SPARSE
  MSCT       = 1 3 4
*/


__device__ void
stencil (float *temp_beta_shared, int iter)
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB


  curandState seed0 = pgpuseed[TperB * blockIdx.x + bidx];	// curand sequence
  LCG_DATATYPE seed1 = pgpuseed1[TperB * blockIdx.x + bidx];    // LCG PRNG sequence


/*
  int rand123_cnt = pcnt[TperB * blockIdx.x + bidx];
  philox4x32_key_t rand123_k = {{bidx, seed1}};
  philox4x32_ctr_t rand123_c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
  union {
    philox4x32_ctr_t c;
    uint4 i;
  } rand123_u;
*/




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
    __syncthreads ();



    for (int i = 0; i < iter; i++) {

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

	  // for profiling purpose
	  //float energy = (n0 ^ n1 ^ n2 ^ n3 ^ n4 ^ n5) & 2;
	  //float val = 0.7;
	  //u_int32_t myrand = 467;
	  //flip |= ((curand_uniform (&myseed) < val) << b);
	  //MSC_DATATYPE flip = n0 ^ n1 ^ n2 ^ n3 ^ n4 ^ n5;


#if 1
	  n0 = MASK_A * ((c >> SHIFT_J0) & 1) ^ n0 ^ c;
	  n1 = MASK_A * ((c >> SHIFT_J1) & 1) ^ n1 ^ c;
	  n2 = MASK_A * ((c >> SHIFT_J2) & 1) ^ n2 ^ c;
	  n3 = MASK_A * ((c >> SHIFT_J3) & 1) ^ n3 ^ c;
	  n4 = MASK_A * ((c >> SHIFT_J4) & 1) ^ n4 ^ c;
	  n5 = MASK_A * ((c >> SHIFT_J5) & 1) ^ n5 ^ c;

#if MSCT == 3
	  // reused a variable, for example n0, is faster than declaring "e"
	  MSC_DATATYPE e =
	    (n0 & MASK_S) +
	    (n1 & MASK_S) +
	    (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
#elif MSCT == 4
	  MSC_DATATYPE e =
	    (n0 & MASK_S) +
	    (n1 & MASK_S) +
	    (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
	  e = (e << 1) + (c & MASK_S);
#endif

	  // process a lattice element
	  MSC_DATATYPE flip = 0;


#if MSCT == 1
	  //#pragma unroll
	  for (int b = 0; b < NBETA_PER_WORD; b++) {
	    int energy =	// range: [0,6]
	      (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
	      (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
	    int spin = c >> b & 1;

            ///*
	    // PROB_GEN == DIRECT_COMPUTE
	    // wrong beta, but performance wise accurate
	    energy = energy * 2 - 6;
	    spin = spin * 2 - 1;
	    PROB_DATATYPE val = expf (2 * energy * temp_prob_shared[b][0]);
	    //PROB_DATATYPE val = __expf (2 * energy * temp_prob_shared[b][0]);
            //*/


	    // PROB_GEN == TABLE_LOOKUP
	    //PROB_DATATYPE val = temp_prob_shared[b][(energy << 1) + spin];

	    
	    //PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]

	    //gpu_lcg (&seed1);
	    //flip |= ((seed1 < val) << b);	// myrand < val ? 1 : 0;


	    flip |= ((myrand < val) << b);	// myrand < val ? 1 : 0;
	  }
#endif


#if MSCT == 3
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    int energy = e >> shift & MASK_E;	// range: [0,6]
	    int spin = c >> shift & 1;
	    PROB_DATATYPE val = temp_prob_shared[shift / MSCT][(energy << 1) + spin];
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }
#endif


#if MSCT == 4
	  /// CURAND

	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    //PROB_DATATYPE myrand = curand_uniform (&seed0);	// range: [0,1]
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }


	  /*
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    PROB_DATATYPE myrand = curand (&seed0);	// range: [0,U_INT32_T_MAX]
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }
	  */


	  
	  /// LCG
	  /*
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    PROB_DATATYPE val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    gpu_lcg (&seed1);

	    //flip |= (((float) seed1 / LCG_M < val) << shift);	// myrand < val ? 1 : 0;
	    flip |= ((seed1 < val) << shift);	// myrand < val ? 1 : 0;
	  }
	  */


	  /// RAND123
	  /*
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    rand123_c.v[0] = (unsigned long) rand123_cnt++;
	    rand123_u.c = philox4x32 (rand123_c, rand123_k);
	    PROB_DATATYPE val, myrand;

	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    myrand = (float) rand123_u.i.x / U_INT32_T_MAX;
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    myrand = (float) rand123_u.i.y / U_INT32_T_MAX;
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    myrand = (float) rand123_u.i.z / U_INT32_T_MAX;
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    myrand = (float) rand123_u.i.w / U_INT32_T_MAX;
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }
	  */

	  /*
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    rand123_c.v[0] = (unsigned long) rand123_cnt++;
	    rand123_u.c = philox4x32 (rand123_c, rand123_k);
	    PROB_DATATYPE val;

	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((rand123_u.i.x < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((rand123_u.i.y < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((rand123_u.i.z < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((rand123_u.i.w < val) << shift);	// myrand < val ? 1 : 0;
	  }
	  */

	  /*
	  for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
	    PROB_DATATYPE val;

	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;

	    shift += MSCT;
	    val = temp_prob_shared[shift >> 2][(e >> shift) & MASK_E];
	    flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	  }
	  */
#endif

#endif



	  l[z][y][x] = c ^ flip;
	}			// z_offset

	__syncthreads ();
      }				// run
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
  pgpuseed1[TperB * blockIdx.x + bidx] = seed1;
  //pcnt[TperB * blockIdx.x + bidx] = rand123_cnt;
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
  // sizeof (u_int32_t) * 16 * 16 * 16 = 16 KB
  __shared__ MSC_DATATYPE l[SZ][SZ][SZ];

  // index for read/write glocal memory
  const int xx = (SZ_HF * (threadIdx.y & 1)) + threadIdx.x;
  const int yy = (SZ_HF * (threadIdx.z & 1)) + (threadIdx.y >> 1);


  // index for reading scratch pad
  const int y = threadIdx.y;
  const int ya = (y + SZ - 1) % SZ;
  const int yb = (y + 1) % SZ;
  const int x0 = (threadIdx.z & 1) ^ (threadIdx.y & 1);	// initial x
  const int x = (threadIdx.x << 1) + x0;
  const int x1 = x + 1 - x0 * 2;
  const int xa = (x + SZ - 1) % SZ;
  const int xb = (x + 1) % SZ;


  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * blockIdx.x + SZ_CUBE * word;

    // lattice scratchpad import
    for (int z_offset = 0; z_offset < SZ; z_offset += (BDz0 >> 1)) {
      int zz = z_offset + (threadIdx.z >> 1);
      l[zz][yy][xx] = plattice[lattice_offset + SZ * SZ * z_offset + bidx];
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


      // \sum J S S = S \sum J S

#if MSCT != 1
      n0 &= MASK_S;
      n1 &= MASK_S;
      n2 &= MASK_S;
      n3 &= MASK_S;
      n4 &= MASK_S;
      n5 &= MASK_S;
      MSC_DATATYPE e = n0 + n1 + n2 + n3 + n4 + n5;
#endif

      //#pragma unroll
      for (int b = 0; b < NBETA_PER_WORD; b++) {

#if MSCT == 1
	int energy =		// range: [0,6]
	  (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
	  (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
#else
	int energy = (e >> (MSCT * b)) & MASK_E;	// range: [0,6]
#endif
	E_shared[b][bidx] += energy;
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
      MSC_DATATYPE c0 = l[z][y][x];
      MSC_DATATYPE c1 = l[z][y][x1];

      for (int b = 0; b < NBETA_PER_WORD; b++){
	E_shared[b][bidx] += (c0 >> (MSCT * b)) & 1;
	E_shared[b][bidx] += (c1 >> (MSCT * b)) & 1;
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


