#include "COPYING"


#ifndef BIT64_H
#define BIT64_H


//#include <stdint.h>



/*
  LENG -           length of an integer
  NBETA -          number of parallel betas
  NBETA_MAX -      align NBETA into "LENG" boundaries
  NBETA_PER_WORD - number of betas combined into a word
  NWORD -          number of words for all betas

  must garantee NBETA <= LENG,
  can overcome this redtriction by extending "lattice1" to multi-word organization

  6 neighbors
  left     (xa,y ,z )    J0
  right    (xb,y ,z )    J1
  up       (x ,ya,z )    J2
  down     (x ,yb,z )    J3
  front    (x ,y ,za)    J4
  back     (x ,y ,zb)    J5
*/




typedef u_int64_t MSC_DATATYPE;
#define LENG 64
#define NBETA_MAX LENG
#define MASK_A  0xffffffffffffffff


////////////////////////////////////////////////////////////////////////
// gpu_stencil_01.cu
// gpu_stencil_11.cu


#if ALLOCATION == SHARED || ALLOCATION == SEPARATED
#if DENSE == COMPACT && MSCT == 4

#define NBETA 56
#define NBETA_PER_WORD 56
#define NWORD 1

#if MSC_FORMAT == 0
#define NSEG_PER_WORD 14
#define NBETA_PER_SEG 4
#define MASK_J  0xfc00000000000000
#define MASK_J0 0x0400000000000000
#define MASK_J1 0x0800000000000000
#define MASK_J2 0x1000000000000000
#define MASK_J3 0x2000000000000000
#define MASK_J4 0x4000000000000000
#define MASK_J5 0x8000000000000000
#define SHIFT_J0 58
#define SHIFT_J1 59
#define SHIFT_J2 60
#define SHIFT_J3 61
#define SHIFT_J4 62
#define SHIFT_J5 63
#define MASK_S  0x0011111111111111
#define MASK_S0 0x00ffffffffffffff
#define MASK_E  0xf
#define SHIFT_MAX 56
#endif
/*
  MASK_J  1111 11-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J0 0000 01-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J1 0000 10-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J2 0001 00-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J3 0010 00-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J4 0100 00-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_J5 1000 00-- 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
  MASK_S  ---- ---- 0001 0001 0001 0001 0001 0001 0001 0001 0001 0001 0001 0001 0001 0001 
  MASK_S0 ---- ---- 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111
  iter0                *    *    *    *    *    *
  iter1               *    *    *    *    *    *
  iter2              *    *    *    *    *    *
  iter3             *    *    *    *    *    *
*/


#endif
#endif



#endif /* BIT64_H */
