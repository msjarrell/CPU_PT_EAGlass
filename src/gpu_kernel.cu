#include "COPYING"




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
kernel_warmup ()
{
  const int bidx = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;	// within a TB

  /// temperature
  // (4 * 32) * 2 = 256 B
  __shared__ float __align__ (32) temp_beta_shared[NBETA];
  if (bidx < NBETA)
    temp_beta_shared[bidx] = ptemp[NBETA_MAX * blockIdx.x + bidx].beta;


  for (int i = 0; i < ITER_WARMUP_KERN; i += ITER_WARMUP_KERNFUNC) {
    stencil (temp_beta_shared, ITER_WARMUP_KERNFUNC);
  }
}




__global__ void
kernel_swap (int rec)
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


  for (int i = 0; i < ITER_SWAP_KERN; i += ITER_SWAP_KERNFUNC) {
    int swap_mod = (i / ITER_SWAP_KERNFUNC) & 1;
    stencil_swap (temp_idx_shared, temp_beta_shared, E, swap_mod);
    stencil (temp_beta_shared, ITER_SWAP_KERNFUNC);
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
  __shared__ double qk_real[3][NBETA];
  __shared__ double qk_imag[3][NBETA];
  __shared__ double qk2_real[6][NBETA];
  __shared__ double qk2_imag[6][NBETA];
  const int lattice_offset0 = SZ_CUBE * (blockIdx.x << 1);
  const int lattice_offset1 = lattice_offset0 + SZ_CUBE;
  const double k=2*PI/SZ;

  for (int offset = 0; offset < SZ_CUBE; offset += TperB) {
    l1[offset + bidx] =		// xord_word 
      plattice1[lattice_offset0 + offset + bidx] ^
      plattice1[lattice_offset1 + offset + bidx];
  }

  __syncthreads();

  // is double an overkill?
  if (bidx < NBETA) {
    float q0 = 0.0f;
    for(int j=0;j<3;j++){
      qk_real[j][bidx] = 0.0f;
      qk_imag[j][bidx] = 0.0f;
    }
    for(int j=0;j<6;j++){
      qk2_real[j][bidx] = 0.0f;
      qk2_imag[j][bidx]= 0.0f;
    }

    MSC_DATATYPE xor_word;
    int xor_bit;

    for (int i = 0; i < SZ_CUBE; i++) {
      xor_word = l1[i];
      xor_bit = (xor_word >> bidx) & 0x1;
      xor_bit = 1 - (xor_bit << 1);	// parallel: +1, reverse: -1

      double bit=xor_bit;
      double x= i % SZ;
      double y= (i / SZ) % SZ;
      double z= (i / SZ) / SZ;
      /*      // 2 * pi / L * x_i
      angel1 = (double) (i % SZ) * 2 * PI / SZ;
      // 2 * pi / L * (x_i + y_i)
      angel2 = (double) (i % SZ + (i / SZ) % SZ) * 2 * PI / SZ;
      */
      q0 += bit;
      /*
      qk_real += (float)xor_bit * cos (angel1);
      qk_imag += (float)xor_bit * sin (angel1);
      qk2_real += (float)xor_bit * cos (angel2);
      qk2_imag += (float)xor_bit * sin (angel2);
      */ 
      qk_real[0][bidx] += bit * cos(x*k);
      qk_real[1][bidx] += bit * cos(y*k);
      qk_real[2][bidx] += bit * cos(z*k);

      qk_imag[0][bidx] += bit * sin(x*k);
      qk_imag[1][bidx] += bit * sin(y*k);
      qk_imag[2][bidx] += bit * sin(z*k);

      qk2_real[0][bidx] += bit * cos(x*k + y*k);
      qk2_real[1][bidx] += bit * cos(x*k - y*k);
      qk2_real[2][bidx] += bit * cos(x*k + z*k);
      qk2_real[3][bidx] += bit * cos(x*k - z*k);
      qk2_real[4][bidx] += bit * cos(y*k + z*k);
      qk2_real[5][bidx] += bit * cos(y*k - z*k);
	         
      qk2_imag[0][bidx] += bit * sin(x*k + y*k);
      qk2_imag[1][bidx] += bit * sin(x*k - y*k);
      qk2_imag[2][bidx] += bit * sin(x*k + z*k);
      qk2_imag[3][bidx] += bit * sin(x*k - z*k);
      qk2_imag[4][bidx] += bit * sin(y*k + z*k);
      qk2_imag[5][bidx] += bit * sin(y*k - z*k);

     

   }

    // save measurements in "st"
    pst[rec].q[blockIdx.x][bidx] = q0;
    for(int j=0;j<3;j++){
      pst[rec].qk_real[j][blockIdx.x][bidx] = qk_real[j][bidx];
      pst[rec].qk_imag[j][blockIdx.x][bidx] = qk_imag[j][bidx];
    }
    //      (float) sqrt (qk_real * qk_real + qk_imag * qk_imag);
    for(int j=0;j<6;j++){
      pst[rec].qk2_real[j][blockIdx.x][bidx] = qk2_real[j][bidx];
      pst[rec].qk2_imag[j][blockIdx.x][bidx] = qk2_imag[j][bidx];
    }
    //      (Float) sqrt (qk2_real * qk2_real + qk2_imag * qk2_imag);
  }
  __syncthreads();
}




