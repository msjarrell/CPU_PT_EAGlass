#include "COPYING"


#ifndef BIT32_H
#define BIT32_H


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




typedef u_int32_t MSC_DATATYPE;
#define LENG 32
#define NBETA_MAX LENG
#define MASK_A  0xffffffff



////////////////////////////////////////////////////////////////////////
// gpu_stencil_00.cu


#if ALLOCATION == SHARED && DENSE == SPARSE && MSCT == 1
#define NBETA 26
#define NBETA_PER_WORD 26
#define NWORD 1
#define MASK_J  0xfc000000
#define MASK_J0 0x04000000
#define MASK_J1 0x08000000
#define MASK_J2 0x10000000
#define MASK_J3 0x20000000
#define MASK_J4 0x40000000
#define MASK_J5 0x80000000
#define SHIFT_J0 26
#define SHIFT_J1 27
#define SHIFT_J2 28
#define SHIFT_J3 29
#define SHIFT_J4 30
#define SHIFT_J5 31
#define MASK_S  0x03ffffff
#define MASK_E  0x1
#define SHIFT_MAX 26
#endif
/*
  MASK_J  1111 1100 0000 0000 0000 0000 0000 0000
  MASK_J0 0000 0100 0000 0000 0000 0000 0000 0000
  MASK_J1 0000 1000 0000 0000 0000 0000 0000 0000
  MASK_J2 0001 0000 0000 0000 0000 0000 0000 0000
  MASK_J3 0010 0000 0000 0000 0000 0000 0000 0000
  MASK_J4 0100 0000 0000 0000 0000 0000 0000 0000
  MASK_J5 1000 0000 0000 0000 0000 0000 0000 0000
  MASK_S  0000 0011 1111 1111 1111 1111 1111 1111
*/




#if ALLOCATION == SHARED && DENSE == SPARSE && MSCT == 3
#define NBETA 30
#define NBETA_PER_WORD 10
#define NWORD 3
#define MASK_J  0x00024924
#define MASK_J0 0x00000004
#define MASK_J1 0x00000020
#define MASK_J2 0x00000100
#define MASK_J3 0x00000800
#define MASK_J4 0x00004000
#define MASK_J5 0x00020000
#define SHIFT_J0 2
#define SHIFT_J1 5
#define SHIFT_J2 8
#define SHIFT_J3 11
#define SHIFT_J4 14
#define SHIFT_J5 17
#define MASK_S  0x09249249
#define MASK_E  0x7
#define SHIFT_MAX 29
#endif
/*
  MASK_J  --00 0000 0000 0010 0100 1001 0010 0100
  MASK_J0 --00 0000 0000 0000 0000 0000 0000 0100
  MASK_J1 --00 0000 0000 0000 0000 0000 0010 0000
  MASK_J2 --00 0000 0000 0000 0000 0001 0000 0000
  MASK_J3 --00 0000 0000 0000 0000 1000 0000 0000
  MASK_J4 --00 0000 0000 0000 0100 0000 0000 0000
  MASK_J5 --00 0000 0000 0010 0000 0000 0000 0000
  MASK_S  --00 1001 0010 0100 1001 0010 0100 1001
*/




#if ALLOCATION == SHARED && DENSE == SPARSE && MSCT == 4
#define NBETA 24
#define NBETA_PER_WORD 8
#define NWORD 3
#define MASK_J  0x00444444
#define MASK_J0 0x00000004
#define MASK_J1 0x00000040
#define MASK_J2 0x00000400
#define MASK_J3 0x00004000
#define MASK_J4 0x00040000
#define MASK_J5 0x00400000
#define SHIFT_J0 2
#define SHIFT_J1 6
#define SHIFT_J2 10
#define SHIFT_J3 14
#define SHIFT_J4 18
#define SHIFT_J5 22
#define MASK_S  0x11111111
#define MASK_E  0xf
#define SHIFT_MAX 32
#endif
/*
  MASK_J  0000 0000 0100 0100 0100 0100 0100 0100
  MASK_J0 0000 0000 0000 0000 0000 0000 0000 0100
  MASK_J1 0000 0000 0000 0000 0000 0000 0100 0000
  MASK_J2 0000 0000 0000 0000 0000 0100 0000 0000
  MASK_J3 0000 0000 0000 0000 0100 0000 0000 0000
  MASK_J4 0000 0000 0000 0100 0000 0000 0000 0000
  MASK_J5 0000 0000 0100 0000 0000 0000 0000 0000
  MASK_S  0001 0001 0001 0001 0001 0001 0001 0001
*/




////////////////////////////////////////////////////////////////////////
// gpu_stencil_01.cu
// gpu_stencil_11.cu


#if ALLOCATION == SHARED || ALLOCATION == SEPARATED
#if DENSE == COMPACT && MSCT == 4

#define NBETA 24
#define NBETA_PER_WORD 24
#define NWORD 1

#if MSC_FORMAT == 0
#define NSEG_PER_WORD 6
#define NBETA_PER_SEG 4
#define MASK_J  0xfc000000
#define MASK_J0 0x04000000
#define MASK_J1 0x08000000
#define MASK_J2 0x10000000
#define MASK_J3 0x20000000
#define MASK_J4 0x40000000
#define MASK_J5 0x80000000
#define SHIFT_J0 26
#define SHIFT_J1 27
#define SHIFT_J2 28
#define SHIFT_J3 29
#define SHIFT_J4 30
#define SHIFT_J5 31
#define MASK_S  0x00111111
#define MASK_S0 0x00ffffff
#define MASK_E  0xf
#define SHIFT_MAX 24
#endif
/*
  MASK_J  1111 11-- 0000 0000 0000 0000 0000 0000
  MASK_J0 0000 01-- 0000 0000 0000 0000 0000 0000
  MASK_J1 0000 10-- 0000 0000 0000 0000 0000 0000
  MASK_J2 0001 00-- 0000 0000 0000 0000 0000 0000
  MASK_J3 0010 00-- 0000 0000 0000 0000 0000 0000
  MASK_J4 0100 00-- 0000 0000 0000 0000 0000 0000
  MASK_J5 1000 00-- 0000 0000 0000 0000 0000 0000
  MASK_S  ---- ---- 0001 0001 0001 0001 0001 0001
  MASK_S0 ---- ---- 1111 1111 1111 1111 1111 1111
  iter0                *    *    *    *    *    *
  iter1               *    *    *    *    *    *
  iter2              *    *    *    *    *    *
  iter3             *    *    *    *    *    *
*/


#if MSC_FORMAT == 1
#define NSEG_PER_WORD 8
#define NBETA_PER_SEG 3
#define MASK_J  0x00888888
#define MASK_J0 0x00000008
#define MASK_J1 0x00000080
#define MASK_J2 0x00000800
#define MASK_J3 0x00008000
#define MASK_J4 0x00080000
#define MASK_J5 0x00800000
#define SHIFT_J0 3
#define SHIFT_J1 7
#define SHIFT_J2 11
#define SHIFT_J3 15
#define SHIFT_J4 19
#define SHIFT_J5 23
#define MASK_S  0x11111111
#define MASK_S0 0x77777777
#define MASK_E  0xf
#define SHIFT_MAX 31
#endif
/*
  MASK_J  0000 0000 1000 1000 1000 1000 1000 1000
  MASK_J0 0000 0000 0000 0000 0000 0000 0000 1000
  MASK_J1 0000 0000 0000 0000 0000 0000 1000 0000
  MASK_J2 0000 0000 0000 0000 0000 1000 0000 0000
  MASK_J3 0000 0000 0000 0000 1000 0000 0000 0000
  MASK_J4 0000 0000 0000 1000 0000 0000 0000 0000
  MASK_J5 0000 0000 1000 0000 0000 0000 0000 0000
  MASK_S  0001 0001 0001 0001 0001 0001 0001 0001
  MASK_S0 0111 0111 0111 0111 0111 0111 0111 0111
  iter0      *    *    *    *    *    *    *    *
  iter1     *    *    *    *    *    *    *    *
  iter2    *    *    *    *    *    *    *    *
*/

#endif
#endif



////////////////////////////////////////////////////////////////////////
// gpu_stencil_21.cu

#if ALLOCATION == INTEGRATED || ALLOCATION == INTEGRATED2
#if DENSE == COMPACT && MSCT == 4

#if MSC_FORMAT == 0
#define NBETA 30
#define NBETA_PER_WORD 10
#define NBETA_PER_SEG 2
#define NWORD 3
#define MASK_J  0x03f00000
#define MASK_J0 0x00100000
#define MASK_J1 0x00200000
#define MASK_J2 0x00400000
#define MASK_J3 0x00800000
#define MASK_J4 0x01000000
#define MASK_J5 0x02000000
#define SHIFT_J0 20
#define SHIFT_J1 21
#define SHIFT_J2 22
#define SHIFT_J3 23
#define SHIFT_J4 24
#define SHIFT_J5 25
#define MASK_S  0x00111111
#define MASK_S0 0x00ffffff
#define MASK_E  0xf
#define SHIFT_MAX 24
#endif
/*
  MASK_J  ---- --11 1111 0000 0000 0000 0000 0000
  MASK_J0 ---- --00 0001 0000 0000 0000 0000 0000
  MASK_J1 ---- --00 0010 0000 0000 0000 0000 0000
  MASK_J2 ---- --00 0100 0000 0000 0000 0000 0000
  MASK_J3 ---- --00 1000 0000 0000 0000 0000 0000
  MASK_J4 ---- --01 0000 0000 0000 0000 0000 0000
  MASK_J5 ---- --10 0000 0000 0000 0000 0000 0000
  MASK_S  ---- ---- ---- 0001 0001 0001 0001 0001
  MASK_S0 ---- ---- ---- 1111 1111 1111 1111 1111
  black0                    *    *    *    *    *
  white0                   *    *    *    *    *
  black1                  *    *    *    *    *
  white1                 *    *    *    *    *
*/
// black(0,0,0): run=0, MASK_JX
// white(0,0,1): run=1, (MASK_JX << 6)


#if MSC_FORMAT == 1
#define NBETA 24
#define NBETA_PER_WORD 8
#define NBETA_PER_SEG 1
#define NWORD 3
#define MASK_J  0x00444444
#define MASK_J0 0x00000004
#define MASK_J1 0x00000040
#define MASK_J2 0x00000400
#define MASK_J3 0x00004000
#define MASK_J4 0x00040000
#define MASK_J5 0x00400000
#define SHIFT_J0 2
#define SHIFT_J1 6
#define SHIFT_J2 10
#define SHIFT_J3 14
#define SHIFT_J4 18
#define SHIFT_J5 22
#define MASK_S  0x11111111
#define MASK_S0 0x33333333
#define MASK_E  0xf
#define SHIFT_MAX 32
#endif
/*
  MASK_J  0000 0000 0100 0100 0100 0100 0100 0100
  MASK_J0 0000 0000 0000 0000 0000 0000 0000 0100
  MASK_J1 0000 0000 0000 0000 0000 0000 0100 0000
  MASK_J2 0000 0000 0000 0000 0000 0100 0000 0000
  MASK_J3 0000 0000 0000 0000 0100 0000 0000 0000
  MASK_J4 0000 0000 0000 0100 0000 0000 0000 0000
  MASK_J5 0000 0000 0100 0000 0000 0000 0000 0000
  MASK_S  0001 0001 0001 0001 0001 0001 0001 0001
  MASK_S0 0011 0011 0011 0011 0011 0011 0011 0011
  black      *    *    *    *    *    *    *    *
  white     *    *    *    *    *    *    *    *
*/
// black(0,0,0): run=0, MASK_JX
// white(0,0,1): run=1, (MASK_JX << 1)


#endif
#endif







//#define SHIFT_MAX 0x02000000
//#define SHIFT_MASK 0x1


#endif /* BIT32_H */
