#include "COPYING"


__device__ void
gpu_init_temp (PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], const int bidx)
{
  if (bidx < NPROB_MAX) {
    for (int b = 0; b < NBETA_PER_WORD; b++) {
      //prob[b][bidx] = 2.0f;
      prob[b][bidx] = U_INT32_T_MAX;
      //prob[b][bidx] = LCG_M + 1;
    }
  }
}



// pre-compute propabilities
// load beta from global memory, compute, save propabilities in shared memory

/*
  bidx        0     1     2     3     4     5     6     7
  energy     -6+H  -6-H  -4+H  -4-H  -2+H  -2-H   0+H   0-H


  even
  energy = bidx - 6 + H
  ---------------------------------------
  bidx        0     2     4     6
  energy     -6+H  -4+H  -2+H   0+H


  odd
  energy = bidx - 7 - H = bidx - 6 + H - (1 + 2H)
  ----------------------------------------------
  bidx        1     3     5     7
  energy     -6-H  -4-H  -2-H   0-H
*/

__device__ void
  gpu_compute_temp
(PROB_DATATYPE prob[NBETA_PER_WORD][NPROB_MAX], float *temp_beta_shared, const int bidx, int word)
{
  // for -2 < H < 2, it is OK to compute only first 8 elements of prob[14]
  // keep the rest unchanged


  if (bidx < NPROB) {
    for (int b = 0; b < NBETA_PER_WORD; b++) {
      float mybeta = temp_beta_shared[NBETA_PER_WORD * word + b];
      //ptemp[NBETA_MAX * blockIdx.x + NBETA_PER_WORD * word + b].beta;
      float energy = bidx - 6 - H - ((-1 * H * 2.0f + 1.0f) * (bidx & 1));

      //prob[b][bidx] = expf (2 * energy * mybeta);
      prob[b][bidx] = expf (2 * energy * mybeta) * U_INT32_T_MAX;
      //prob[b][bidx] = expf (2 * energy * mybeta) * LCG_M;
    }
  }

}





// propose a temperature shuffle (inside a lattice)
__device__ void
gpu_shuffle (int *temp_idx, float *temp_beta, float *E, const int bidx, int mod)
{

  int idx0, idx1, bidx_max;
  float delta_E, delta_beta;
  float myrand, val;
  int tmp0;
  float tmp1;


  curandState seed0 = pgpuseed[TperB * blockIdx.x + bidx];	// needed by curand
  // LCG_DATATYPE seed1 = pgpuseed1[TperB * blockIdx.x + bidx];
  /*
  int rand123_cnt = pcnt[TperB * blockIdx.x + bidx];
  philox4x32_key_t rand123_k = {{bidx, seed1}};
  philox4x32_ctr_t rand123_c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
  union {
    philox4x32_ctr_t c;
    uint4 i;
  } rand123_u;
  
  */

  __shared__ int __align__ (32) order[NBETA];

  if (bidx < NBETA)
    order[temp_idx[bidx]] = bidx;
  __syncthreads ();



  /*
     #ifdef DEBUG_PRINT_E
     if (blockIdx.x == 0 && bidx == 0) {
     for (int b = 0; b < NBETA; b++) {
     printf ("9 b=%02d %f \tE = %d\n", b, temp_beta[b], E[b]);
     }
     }
     #endif
   */



  if (mod == 0) {
    // 0 swap 1 , 2 swap 3 , 4 swap 5 , ...
    bidx_max = NBETA / 2;	// boundary
    idx0 = order[bidx << 1];
    idx1 = order[(bidx << 1) + 1];
  }
  else if (mod == 1) {
    // 0 , 1 swap 2 , 3 swap 4 , ...
    bidx_max = NBETA / 2 - 1;
    idx0 = order[(bidx << 1) + 1];
    idx1 = order[(bidx << 1) + 2];
  }

  if (bidx < bidx_max) {
    myrand = curand_uniform (&seed0);	// range: [0,1]
    /*
    rand123_c.v[0] = (unsigned long) rand123_cnt++;
    rand123_u.c = philox4x32 (rand123_c, rand123_k);
    myrand = (float) rand123_u.i.x / U_INT32_T_MAX;
    */
    delta_E = E[idx0] - E[idx1];
    delta_beta = temp_beta[idx0] - temp_beta[idx1];
    // test "expf" and "__expf" with extremely large input
    // verified their compatibities with positive infinity representation
    val = expf (delta_E * delta_beta);


    /*
       printf ("swap probability: %f =  exp (%f * %f) \t beta[%02d] = %f, beta[%02d] = %f \n",
       val, delta_E, delta_beta,
       idx0, temp_beta[idx0], idx1, temp_beta[idx1]);
     */

    // swap
    // branch would not hurt performance
    if (myrand < val) {
      tmp0 = temp_idx[idx0];
      temp_idx[idx0] = temp_idx[idx1];
      temp_idx[idx1] = tmp0;

      tmp1 = temp_beta[idx0];
      temp_beta[idx0] = temp_beta[idx1];
      temp_beta[idx1] = tmp1;
    }
  }

  pgpuseed[TperB * blockIdx.x + bidx] = seed0;
  //pcnt[TperB * blockIdx.x + bidx] = rand123_cnt;
  __syncthreads ();
}






__device__ void
//__forceinline__
gpu_reduction (float *a, short a_shared[NBETA_PER_WORD][TperB], const int bidx,
	       int word)
{
  // skewed sequential reduction is faster than tree reduction

#if 1
  // multi-threaded sequential reduction

  if (bidx < NBETA_PER_WORD) {
    int aaa = 0;
    for (int t = 0; t < TperB; t++) {
      // skew loop iteration from "t" to "(t + bidx) % TperB"
      // to avoid shared memory bank confict
      aaa += a_shared[bidx][(t + bidx) % TperB];
    }

    // save the summation
    a[NBETA_PER_WORD * word + bidx] = aaa;
  }
#endif



#if 0
  // tree reduction

  __syncthreads ();

  //int powerof2 = power2floor (TperB);

  for (int b = 0; b < NBETA_PER_WORD; b++) {
    /*
    for (int stride = TperB / 2; stride >= 1; stride >>= 1) {
      if (bidx < stride)
	a_shared[b][bidx] += a_shared[b][stride + bidx];
      __syncthreads ();
    }
    */

    if (bidx < TperB - powerof2) {
      a_shared[b][bidx] += a_shared[b][powerof2 + bidx];
    }
    for (int stride = powerof2 / 2; stride >= 1; stride >>= 1) {
      if (bidx < stride)
	a_shared[b][bidx] += a_shared[b][stride + bidx];
      __syncthreads ();
    }

  }

  // save the summation
  if (bidx < NBETA_PER_WORD)
    a[NBETA_PER_WORD * word + bidx] = (float) a_shared[bidx][0];
#endif

}
