#include "COPYING"


#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

//#include "cuda_util.cuh"
#include "include_cuda/cutil_yeah.h"

#include "sim.h"
#include "sim.cuh"
#include "lcg.h"
#include "host_kernel.h"


void
host_launcher (float beta_low, float beta_high, char *mydir, int node,
	       int device)
{
  // select a GPU device
  cudaSetDevice (device);

  // configure the GPU SRAM
  //cudaFuncSetCacheConfig (kernel_unified, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig (kernel_warmup, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig (kernel_swap, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig (kernel_rearrange, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig (kernel_compute_q, cudaFuncCachePreferShared);


  int dim_grid0 = GD;
  dim3 dim_block0 (BDx0, BDy0, BDz0);

  int dim_grid3 = GD;
  int dim_block3 = BDx3;

  int dim_grid4 = GD_HF;
  int dim_block4 = BDx3;


  // initilize random sequence
  srand (time (NULL) + 10 * node + device);


  // gpuseed - curand seed
  int myrand = rand ();
  CUDAMEMCPYTOSYMBOL (seed, &myrand, int);

  curandState *gpuseed_dev;
  size_t gpuseed_sz = sizeof (curandState) * TperB * GD;
  CUDAMALLOC (gpuseed_dev, gpuseed_sz, curandState *);
  CUDAMEMCPYTOSYMBOL (pgpuseed, &gpuseed_dev, curandState *);

  // how often should we re-initialize gpuseed???
  CUDAKERNELSYNC (kernel_init_seed, dim_grid3, dim_block3);


 
 // seeds for LCG PRNG
  LCG_DATATYPE *gpuseed1, *gpuseed1_dev;
  size_t gpuseed1_sz = sizeof (LCG_DATATYPE) * TperB * GD;
  gpuseed1 = (LCG_DATATYPE *) malloc (gpuseed1_sz);
  CUDAMALLOC (gpuseed1_dev, gpuseed1_sz, LCG_DATATYPE *);
  CUDAMEMCPYTOSYMBOL (pgpuseed1, &gpuseed1_dev, LCG_DATATYPE *);

  host_init_seed (gpuseed1);
  CUDAMEMCPY (gpuseed1_dev, gpuseed1, gpuseed1_sz, cudaMemcpyHostToDevice);



  // cnt - counter for rand123 PRNG
  // ziter * 2 * NBETA * ITER ~= 540 * ITER ~= 1 * 10^9
  int *cnt, *cnt_dev;
  size_t cnt_sz = sizeof (int) * TperB * GD;
  cnt = (int *) malloc (cnt_sz);
  CUDAMALLOC (cnt_dev, cnt_sz, int *);
  CUDAMEMCPYTOSYMBOL (pcnt, &cnt_dev, int *);
  for (int i = 0; i < TperB * GD; i++)
    cnt[i] = 0;
  CUDAMEMCPY (cnt_dev, cnt, cnt_sz, cudaMemcpyHostToDevice);



  // lattice
  MSC_DATATYPE *lattice, *lattice_dev;
  size_t lattice_sz = sizeof (MSC_DATATYPE) * SZ_CUBE * NWORD * GD;
  lattice = (MSC_DATATYPE *) malloc (lattice_sz);
  CUDAMALLOC (lattice_dev, lattice_sz, MSC_DATATYPE *);
  host_init_lattice (lattice);
  CUDAMEMCPY (lattice_dev, lattice, lattice_sz, cudaMemcpyHostToDevice);
  CUDAMEMCPYTOSYMBOL (plattice, &lattice_dev, MSC_DATATYPE *);

  // lattice1
  // spins have been rearranged to reflect the temperature order
  MSC_DATATYPE *lattice1_dev;
  size_t lattice1_sz = sizeof (MSC_DATATYPE) * SZ_CUBE * GD;
  CUDAMALLOC (lattice1_dev, lattice1_sz, MSC_DATATYPE *);
  CUDAMEMCPYTOSYMBOL (plattice1, &lattice1_dev, MSC_DATATYPE *);

  // temp - index and beta
  Temp *temp, *temp_dev;
  size_t temp_sz = sizeof (Temp) * NBETA_MAX * GD;
  temp = (Temp *) malloc (temp_sz);
  CUDAMALLOC (temp_dev, temp_sz, Temp *);
  host_init_temp (temp, beta_low, beta_high);
  CUDAMEMCPY (temp_dev, temp, temp_sz, cudaMemcpyHostToDevice);
  CUDAMEMCPYTOSYMBOL (ptemp, &temp_dev, Temp *);

  // st - status records
  St *st, *st_dev;
  size_t st_sz = sizeof (St) * ITER_SWAP / ITER_SWAP_KERN;
  st = (St *) malloc (st_sz);
  CUDAMALLOC (st_dev, st_sz, St *);
  CUDAMEMCPYTOSYMBOL (pst, &st_dev, St *);
#ifdef DEBUG0
  printf ("st_sz = %f MB\n", (float) st_sz / 1024 / 1024);
#endif



  char event[STR_LENG];

  //Timing t[4];
  //host_timing_init (t, 4);

  double t[4][2];
  double t2[3];
  double t3 = 0;

  putchar ('\n');
  //host_report_speed_title ();


#if 1
  /// GPU ising

  // warm up runs
  t2[0] = host_time_now ();
  for (int i = 0; i < ITER_WARMUP; i += ITER_WARMUP_KERN) {
    t[0][0] = host_time_now ();

    CUDAKERNELSYNC (kernel_warmup, dim_grid0, dim_block0);
    //CUDAKERNELSYNC (kernel_unified, dim_grid0, dim_block0, 0, 0);

    t[0][1] = host_time_now ();
    //sprintf (event, "n%03d d%d warmup %8d/%08d", node, device, i, ITER_WARMUP);
    //host_report_speed (t[0][0], t[0][1], ITER_WARMUP_KERN, event);
  }

  t2[1] = host_time_now ();

  // swap runs
  for (int i = 0; i < ITER_SWAP; i += ITER_SWAP_KERN) {
    t[1][0] = host_time_now ();

    CUDAKERNELSYNC (kernel_swap, dim_grid0, dim_block0, i / ITER_SWAP_KERN);
    //CUDAKERNELSYNC (kernel_unified, dim_grid0, dim_block0, i / ITER_SWAP_KERN, 1);

    t[1][1] = host_time_now ();
    t3 += t[1][1] - t[1][0];

    CUDAKERNELSYNC (kernel_rearrange, dim_grid3, dim_block3);
    CUDAKERNELSYNC (kernel_compute_q, dim_grid4, dim_block4, i / ITER_SWAP_KERN);

    t[2][1] = host_time_now ();

    //sprintf (event, "n%03d d%d PT     %8d/%08d", node, device, i, ITER_SWAP);
    //host_report_speed (t[1][0], t[1][1], ITER_SWAP_KERN, event);
  }
  t2[2] = host_time_now ();
#endif



#if 0
  /// CPU ising

  // warm up runs
  t2[0] = host_time_now ();
  for (int i = 0; i < ITER_WARMUP; i += ITER_WARMUP_KERN) {
    t[0][0] = host_time_now ();
    host_kernel_warmup (lattice, temp);
    t[0][1] = host_time_now ();
    sprintf (event, "n%03d d%d warmup %8d/%08d", node, device, i, ITER_WARMUP);
    host_report_speed (t[0][0], t[0][1], ITER_WARMUP_KERN, event);
  }
  
  t2[1] = host_time_now ();
  
  // swap runs
  for (int i = 0; i < ITER_SWAP; i += ITER_SWAP_KERN) {
    t[1][0] = host_time_now ();
    host_kernel_swap (lattice, temp, st, i / ITER_SWAP_KERN);
    t[1][1] = host_time_now ();
    t3 += t[1][1] - t[1][0];
    sprintf (event, "n%03d d%d PT     %8d/%08d", node, device, i, ITER_SWAP);
    host_report_speed (t[1][0], t[1][1], ITER_SWAP_KERN, event);
  }
  t2[1] = host_time_now ();
#endif




#ifndef NO_OUTPUT
  CUDAMEMCPY (st, st_dev, st_sz, cudaMemcpyDeviceToHost);
  host_save_st (st, mydir, node, device);
#endif



  // report overall speed
  putchar ('\n');
  sprintf (event, "n%03d d%d overall warmup          ", node, device);
  host_report_speed (t2[0], t2[1], ITER_WARMUP, event);
  sprintf (event, "n%03d d%d overall PT (no measure) ", node, device);
  host_report_speed (0, t3, ITER_SWAP, event);
  sprintf (event, "n%03d d%d overall PT              ", node, device);
  host_report_speed (t2[1], t2[2], ITER_SWAP, event);
  sprintf (event, "n%03d d%d overall simulation      ", node, device);
  host_report_speed (t2[0], t2[2], ITER_WARMUP + ITER_SWAP, event);
  putchar ('\n');


  CUDAFREE (gpuseed_dev);
  CUDAFREE (gpuseed1_dev);
  CUDAFREE (cnt_dev);
  CUDAFREE (lattice_dev);
  CUDAFREE (lattice1_dev);
  CUDAFREE (temp_dev);
  CUDAFREE (st_dev);

  free (gpuseed1);
  free (cnt);
  free (lattice);
  free (temp);
  free (st);
}

