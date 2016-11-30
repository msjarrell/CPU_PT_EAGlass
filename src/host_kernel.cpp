/*
  PATTERN = CHECKERBOARD
  DENSE   = SPARSE
  MSCT    = 3 4
*/


// a performance reference


#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

//#include <cuda.h>
//#include <curand.h>
//#include <curand_kernel.h>

#include "sim.h"
//#include "sim.cuh"





void
host_stencil (MSC_DATATYPE * lattice, float *temp_beta_shared, int iter)
{
  /// temperature scratchpad
  float temp_prob_shared[NBETA_PER_WORD][NPROB_MAX];

  for (int b = 0; b < NBETA_PER_WORD; b++)
    for (int bidx = 0; bidx < NPROB_MAX; bidx++)
      temp_prob_shared[b][bidx] = 2;


  /// lattice scratchpad
  MSC_DATATYPE l[SZ][SZ][SZ];


  for (int block = 0; block < GD; block++) {
    for (int word = 0; word < NWORD; word++) {
      int lattice_offset = SZ_CUBE * NWORD * block + SZ_CUBE * word;

      // initilize temperature scatchpad
      for (int bidx = 0; bidx < 8; bidx++) {
	for (int b = 0; b < NBETA_PER_WORD; b++) {
	  float mybeta = temp_beta_shared[NBETA_PER_WORD * word + b];
	  float energy = bidx - 6 + H - ((H * 2.0f + 1.0f) * (bidx & 1));
	  temp_prob_shared[b][bidx] = expf (2 * energy * mybeta);
	}
      }


      // lattice scratchpad import
      for (int z = 0; z < SZ; z++)
	for (int y = 0; y < SZ; y++)
	  for (int x = 0; x < SZ; x++)
	    l[z][y][x] =
	      lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x];


      for (int i = 0; i < iter; i++) {
	for (int run = 0; run < 2; run++) {

	  for (int z = 0; z < SZ; z++) {
	    int za = (z == 0) ? (SZ - 1) : (z - 1);
	    int zb = (z == (SZ - 1)) ? 0 : (z + 1);

	    for (int y = 0; y < SZ; y++) {
	      int ya = (y == 0) ? (SZ - 1) : (y - 1);
	      int yb = (y == (SZ - 1)) ? 0 : (y + 1);

	      for (int xi = 0; xi < SZ_HF; xi++) {
		int x0 = (z & 1) ^ (y & 1) ^ run;	// initial x
		int x = (xi << 1) + x0;
		int xa = (x == 0) ? (SZ - 1) : (x - 1);
		int xb = (x == (SZ - 1)) ? 0 : (x + 1);

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

#if MSCT == 3
		MSC_DATATYPE h =
		  (n0 & MASK_S) +
		  (n1 & MASK_S) +
		  (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) +
		  (n5 & MASK_S);
#elif MSCT == 4
		MSC_DATATYPE h =
		  (n0 & MASK_S) +
		  (n1 & MASK_S) +
		  (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) +
		  (n5 & MASK_S);
		h = (h << 1) + (c & MASK_S);
#endif


		MSC_DATATYPE flip = 0;

#if MSCT == 1
		//#pragma unroll
		for (int b = 0; b < NBETA_PER_WORD; b++) {
		  int energy =	// range: [0,6]
		    (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
		    (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
		  int spin = c >> b & 1;
		  float val = temp_prob_shared[b][(energy << 1) + spin];
		  float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		  flip |= ((myrand < val) << b);	// myrand < val ? 1 : 0;
		}
#endif

#if MSCT == 3
		for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
		  int energy = h >> shift & MASK_E;	// range: [0,6]
		  int spin = c >> shift & 1;
		  float val =
		    temp_prob_shared[shift / MSCT][(energy << 1) + spin];
		  float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		  flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
		}
#endif

#if MSCT == 4
		for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
		  float val =
		    temp_prob_shared[shift >> 2][(h >> shift) & MASK_E];
		  float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		  flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
		}
#endif

		l[z][y][x] = c ^ flip;

	      }			// xi 
	    }			// y
	  }			// z
	}			// run
      }				// iter


      // lattice scratchpad export
      for (int z = 0; z < SZ; z++)
	for (int y = 0; y < SZ; y++)
	  for (int x = 0; x < SZ; x++)
	    lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x] =
	      l[z][y][x];

    }				// word
  }				// block

}




/*
void
host_stencil_sequential (MSC_DATATYPE * lattice, Temp * temp, int iter)
{
  /// temperature scratchpad
  float temp_prob_shared[NBETA_PER_WORD][NPROB_MAX];

  for (int b = 0; b < NBETA_PER_WORD; b++)
    for (int bidx = 0; bidx < NPROB_MAX; bidx++)
      temp_prob_shared[b][bidx] = 2;


  /// lattice scratchpad
  MSC_DATATYPE l[SZ][SZ][SZ];




  for (int block = 0; block < GD; block++) {
    for (int word = 0; word < NWORD; word++) {
      int lattice_offset = SZ_CUBE * NWORD * block + SZ_CUBE * word;

      // initilize temperature scatchpad
      for (int b = 0; b < NBETA_PER_WORD; b++) {
	for (int bidx = 0; bidx < 8; bidx++) {
	  float mybeta =
	    temp[NBETA_MAX * block + NBETA_PER_WORD * word + b].beta;
	  float energy = bidx - 6 + H - ((H * 2.0f + 1.0f) * (bidx & 1));
	  temp_prob_shared[b][bidx] = expf (2 * energy * mybeta);
	}
      }


      // lattice scratchpad import
      for (int z = 0; z < SZ; z++)
	for (int y = 0; y < SZ; y++)
	  for (int x = 0; x < SZ; x++)
	    l[z][y][x] =
	      lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x];


      for (int i = 0; i < iter; i++) {

	for (int z = 0; z < SZ; z++) {
	  int za = (z == 0) ? (SZ - 1) : (z - 1);
	  int zb = (z == (SZ - 1)) ? 0 : (z + 1);

	  for (int y = 0; y < SZ; y++) {
	    int ya = (y == 0) ? (SZ - 1) : (y - 1);
	    int yb = (y == (SZ - 1)) ? 0 : (y + 1);

	    for (int x = 0; x < SZ; x++) {
	      int xa = (x == 0) ? (SZ - 1) : (x - 1);
	      int xb = (x == (SZ - 1)) ? 0 : (x + 1);

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

#if MSCT == 3
	      MSC_DATATYPE h =
		(n0 & MASK_S) +
		(n1 & MASK_S) +
		(n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
#elif MSCT == 4
	      MSC_DATATYPE h =
		(n0 & MASK_S) +
		(n1 & MASK_S) +
		(n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
	      h = (h << 1) + (c & MASK_S);
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
		float val = temp_prob_shared[b][(energy << 1) + spin];
		float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		flip |= ((myrand < val) << b);	// myrand < val ? 1 : 0;
	      }
#endif

#if MSCT == 3
	      for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
		int energy = h >> shift & MASK_E;	// range: [0,6]
		int spin = c >> shift & 1;
		float val =
		  temp_prob_shared[shift / MSCT][(energy << 1) + spin];
		float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	      }
#endif

#if MSCT == 4
	      for (int shift = 0; shift < SHIFT_MAX; shift += MSCT) {
		float val =
		  temp_prob_shared[shift >> 2][(h >> shift) & MASK_E];
		float myrand = (float) rand () / RAND_MAX;	// range: [0,1]
		flip |= ((myrand < val) << shift);	// myrand < val ? 1 : 0;
	      }
#endif

	      l[z][y][x] = c ^ flip;

	    }			// xi 
	  }			// y
	}			// z
      }				// iter



      // lattice scratchpad export
      for (int z = 0; z < SZ; z++)
	for (int y = 0; y < SZ; y++)
	  for (int x = 0; x < SZ; x++)
	    lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x] =
	      l[z][y][x];

    }				// word
  }				// block
}
*/





void
host_stencil_swap (MSC_DATATYPE * lattice, int *E)
{

  int E_shared[NBETA_PER_WORD];

  /// lattice scratchpad
  MSC_DATATYPE l[SZ][SZ][SZ];


  //for (int block = 0; block < GD; block++) {
  int block = 0;
  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * block + SZ_CUBE * word;

    // lattice scratchpad import
    for (int z = 0; z < SZ; z++)
      for (int y = 0; y < SZ; y++)
	for (int x = 0; x < SZ; x++)
	  l[z][y][x] = lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x];

    // reset energy
    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b] = 0;


    for (int z = 0; z < SZ; z++) {
      int za = (z == 0) ? (SZ - 1) : (z - 1);
      int zb = (z == (SZ - 1)) ? 0 : (z + 1);

      for (int y = 0; y < SZ; y++) {
	int ya = (y == 0) ? (SZ - 1) : (y - 1);
	int yb = (y == (SZ - 1)) ? 0 : (y + 1);


	for (int xi = 0; xi < SZ_HF; xi++) {
	  int x0 = (z & 1) ^ (y & 1);	// initial x
	  int x = (xi << 1) + x0;
	  int xa = (x == 0) ? (SZ - 1) : (x - 1);
	  int xb = (x == (SZ - 1)) ? 0 : (x + 1);

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

#if MSCT != 1
	  MSC_DATATYPE h =
	    (n0 & MASK_S) +
	    (n1 & MASK_S) +
	    (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
#endif


	  for (int b = 0; b < NBETA_PER_WORD; b++) {
#if MSCT == 1
	    int energy =	// range: [0,6]
	      (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
	      (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
#else
	    int energy = (h >> (MSCT * b)) & MASK_E;	// range: [0,6]
#endif
	    energy = energy * 2 - 6;
	    E_shared[b] += energy;
	  }


	}			// xi 
      }				// y
    }				// z

    for (int b = 0; b < NBETA_PER_WORD; b++)
      E[NBETA_PER_WORD * word + b] += E_shared[b];

  }				// word

  //}                           // block
}




/*
void
host_stencil_swap_sequential (MSC_DATATYPE * lattice, double *eee)
{

  int E_shared[NBETA_PER_WORD];

  /// lattice scratchpad
  MSC_DATATYPE l[SZ][SZ][SZ];


  //for (int block = 0; block < GD; block++) {
  int block = 0;
  for (int word = 0; word < NWORD; word++) {
    int lattice_offset = SZ_CUBE * NWORD * block + SZ_CUBE * word;

    // lattice scratchpad import
    for (int z = 0; z < SZ; z++)
      for (int y = 0; y < SZ; y++)
	for (int x = 0; x < SZ; x++)
	  l[z][y][x] = lattice[lattice_offset + (SZ * SZ * z) + (SZ * y) + x];

    // reset energy
    for (int b = 0; b < NBETA_PER_WORD; b++)
      E_shared[b] = 0;


    for (int z = 0; z < SZ; z++) {
      int za = (z == 0) ? (SZ - 1) : (z - 1);
      int zb = (z == (SZ - 1)) ? 0 : (z + 1);

      for (int y = 0; y < SZ; y++) {
	int ya = (y == 0) ? (SZ - 1) : (y - 1);
	int yb = (y == (SZ - 1)) ? 0 : (y + 1);

	for (int x = 0; x < SZ; x++) {
	  int xa = (x == 0) ? (SZ - 1) : (x - 1);
	  int xb = (x == (SZ - 1)) ? 0 : (x + 1);

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

#if MSCT != 1
	  MSC_DATATYPE h =
	    (n0 & MASK_S) +
	    (n1 & MASK_S) +
	    (n2 & MASK_S) + (n3 & MASK_S) + (n4 & MASK_S) + (n5 & MASK_S);
#endif


	  for (int b = 0; b < NBETA_PER_WORD; b++) {
#if MSCT == 1
	    int energy =	// range: [0,6]
	      (n0 >> b & 1) + (n1 >> b & 1) + (n2 >> b & 1) +
	      (n3 >> b & 1) + (n4 >> b & 1) + (n5 >> b & 1);
#else
	    int energy = (h >> (MSCT * b)) & MASK_E;	// range: [0,6]
#endif
	    //energy = energy * 2 - 6;
	    energy -= 3;
	    E_shared[b] += energy;
	  }


	}			// xi 
      }				// y
    }				// z


    for (int b = 0; b < NBETA_PER_WORD; b++)
      eee[NBETA_PER_WORD * word + b] += (double) E_shared[b];

  }				// word

  //}                           // block
}
*/






/*
  // host_launcher.cu

  // random table
  float *randtable, *randtable_dev;
  // 4 * 512 * 32 * 2 * 1000 = 2K * 64 * 1000 = 128M
  size_t randtable_sz = sizeof (float) * SZ_CUBE * NBETA_MAX * GD * ITER_KERN;
  randtable = (float *) malloc (randtable_sz);
  cudaMalloc ((void **) &randtable_dev, randtable_sz);
  cudaMemcpyToSymbol ("prand", &randtable_dev, sizeof (float *), 0, cudaMemcpyHostToDevice);

  free (randtable);
  cudaFree (randtable_dev);
 */



/*
//kernel_init_randtable <<< dim_grid3, dim_block3 >>> (gpuseed_dev, randtable_dev);
void
kernel_init_randtable (curandState * gpuseed, float * randtable)
{
  const int bidx = threadIdx.x;
  curandState myseed = gpuseed[TperB * blockIdx.x + bidx];	// needed by curand

  int bound = SZ_CUBE * NBETA_MAX * ITER_KERN;
  int offset = bound * blockIdx.x;

  for (int i = 0; i < bound; i+= TperB)
    //randtable [offset + i + bidx] = curand_uniform (&myseed);	// range: [0,1]
  randtable [offset + i + bidx] = curand_uniform (gpuseed);	// range: [0,1]

}
*/


 /*
    //host_init_randtable (randtable);
    void
    host_init_randtable (float *randtable)
    {
    int bound = SZ_CUBE * NBETA_MAX * GD * ITER_KERN;
    for (int i = 0; i < bound; i++)
    randtable[i] = (float) rand () / RAND_MAX;
    }
  */





void
host_kernel_warmup (MSC_DATATYPE * lattice, Temp * temp)
{
  float temp_beta_shared[NBETA];

  for (int bidx = 0; bidx < NBETA; bidx++)
    temp_beta_shared[bidx] = temp[NBETA_MAX * 0 + bidx].beta;

  for (int i = 0; i < ITER_WARMUP_KERN; i += ITER_WARMUP_KERNFUNC)
    host_stencil (lattice, temp_beta_shared, ITER_WARMUP_KERNFUNC);

}


void
host_kernel_swap (MSC_DATATYPE * lattice, Temp * temp, St * st, int rec)
{
  int temp_idx_shared[NBETA];
  float temp_beta_shared[NBETA];
  int E[NBETA];

  int swap_mod;

  // load temperature
  for (int bidx = 0; bidx < NBETA; bidx++) {
    temp_idx_shared[bidx] = temp[NBETA_MAX * 0 + bidx].idx;
    temp_beta_shared[bidx] = temp[NBETA_MAX * 0 + bidx].beta;
  }

  for (int i = 0; i < ITER_SWAP_KERN; i += ITER_SWAP_KERNFUNC) {
    swap_mod = (i / ITER_SWAP_KERNFUNC) & 1;
    host_stencil_swap (lattice, E);
    host_stencil (lattice, temp_beta_shared, ITER_SWAP_KERNFUNC);
  }


  // store temperature
  for (int bidx = 0; bidx < NBETA; bidx++) {
    temp[NBETA_MAX * 0 + bidx].idx = temp_idx_shared[bidx];
    temp[NBETA_MAX * 0 + bidx].beta = temp_beta_shared[bidx];
  }

  /*
  // store energy status
  for (int bidx = 0; bidx < NBETA; bidx++)
    st[rec].e[0][temp_idx_shared[bidx]] = E[bidx];
  */
}




