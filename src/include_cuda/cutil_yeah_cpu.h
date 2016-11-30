/*
 * implement some CUDA APIs in C stdlib
 * to help write universal code for both CPU/GPU
 */


#ifndef _CUTIL_YEAH_CPU_
#define _CUTIL_YEAH_CPU_



#define cudaFuncSetCacheConfig(...) \
  ;

#define cudaSetDevice(...) \
  ;

#define cudaGetLastError() \
  ;



#define CUDAMALLOC(ptr, sz, type) \
  ptr = (type) malloc (sz)

#define CUDAFREE(...) \
  free (__VA_ARGS__);

#define CUDAMEMCPY(dst, src, sz, direction) \
  memcpy (dst, src, sz);

#define CUDAMEMCPYTOSYMBOL(dst, src, type)\
  dst = *src;

#define CUDAKERNEL(funcname, dim_grid, dim_block, ...) \
  funcname (__VA_ARGS__);





#endif



