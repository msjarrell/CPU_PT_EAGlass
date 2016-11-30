#include "COPYING"


#ifndef PARAMETERS_H
#define PARAMETERS_H



#define BETA_LOW 0.1f
#define BETA_HIGH 1.8f

// external field
#define H 0.1f
//#define H 1.0f

// randomly initialize J
#define RANDJ
#define RANDS


// using the same random number for every spins integrated in a world
//#define SHARERAND







// iteration parameters


// total monte-carlo sweeps: 2,000,000
// status samples:           absolutely no more than 10,000, tipically 1,000

// for one realization, each status sample consumes memory:
//   (NBETA_MAX * sizeof (int)) * 2 = 256B
// assume assign 32 realiztions on a GPU, memory for saving status:
//   256 * 32 * 10,000 = 64MB

// simulation time estimzation for 16 realizations, 16^3 cubic lattice
// NBETA * (16 ^ 3) * 16 * (2 * 10^6) * (50PS/spin) = 170 seconds


///*
#define ITER_WARMUP          4000
#define ITER_WARMUP_KERN     1000
#define ITER_WARMUP_KERNFUNC 200
#define ITER_SWAP            4000
#define ITER_SWAP_KERN       1000
#define ITER_SWAP_KERNFUNC   10
//*/



#define REC_SIZE ( ITER_SWAP / ITER_SWAP_KERN )





// lattice size
// SZ must be even
// SZz must divides SZ

#define SZ 16
#define SZz SZ

#define SZ_HF (SZ / 2)
#define SZ_CUBE (SZ * SZ * SZz)
#define SZ_CUBE_HF (SZ_CUBE / 2)
#define SZ_TILE (SZ * SZ * SZ)


// SM per GPU
// should implement GD = func (prop.multiProcessorCount);

// GD - blocksPerGrid, must be even
// BD - threadsPerBlock
// when modifing "GD", should also update "GD_HF",

#define GD 32
#define GD_HF 16

#define TperB 256
// checkerboard 3D block
#define BDx0 SZ_HF
#define BDy0 SZ
#define BDz0 2
// 1D block
#define BDx3 TperB//(SZ * SZ * 1)
#define BDy3 1
#define BDz3 1


#define MPIRANK_PER_NODE 2

#endif /* PARAMETERS_H */

