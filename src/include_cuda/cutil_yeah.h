/*

cuda notes

1.
CE1 (cudaAPI (...));
CE2 (cudaAPI (...), "Error message");

2.
kernel <<< ... >>> (...);
cutilCheckMsgAndSync ("kernel failed\n");


// call this function before the first kernal invocation
cudaGetLastError ();

*/




#ifndef _CUTIL_YEAH_
#define _CUTIL_YEAH_

#define CUDAVERSION 4





// NVIDIA_CUDA_SDK/C/common/inc/cutil_inline.h
// this library has been eliminated since CUDA 5
#include "cutil_inline.h"






// CUDA API Error-Checking Wrapper
inline void CE1(const cudaError_t rv)
{
  if (rv != cudaSuccess) {
    printf("CUDA error %d, %s.\n", rv, cudaGetErrorString (rv));
    exit (1);
  }
}


// CUDA API Error-Checking Wrapper
inline void CE2(const cudaError_t rv, char* pMsg)
{
  if (rv != cudaSuccess) {
    printf("CUDA error %d on %s, %s.\n", rv, pMsg, cudaGetErrorString (rv));
    cutilCheckMsg(pMsg);
    exit (1);
  }
}






#define CUDAMALLOC(ptr, sz, type) \
  cudaMalloc ((void **) &ptr, sz)

#define CUDAFREE(...) \
  cudaFree (__VA_ARGS__)

#define CUDAMEMCPY(dst, src, sz, direction) \
  CE2 (cudaMemcpy (dst, src, sz, direction), #dst)


#if CUDAVERSION == 4
#define CUDAMEMCPYTOSYMBOL(dst, src, type) \
  CE2 (cudaMemcpyToSymbol (#dst, src, sizeof (type), 0, cudaMemcpyHostToDevice), #dst)
#elif CUDAVERSION == 5
#define CUDAMEMCPYTOSYMBOL(dst, src, type) \
  CE2 (cudaMemcpyToSymbol (dst, src, sizeof (type), 0, cudaMemcpyHostToDevice), #dst)
#else
  printf ("cuda version not supportted by this tool set\n");
#endif



// call "cutilDeviceSynchronize" after each kernel
#define CUDAKERNELSYNC(funcname, dim_grid, dim_block, ...) \
  funcname <<< dim_grid, dim_block >>> (__VA_ARGS__); \
  cutilCheckMsgAndSync ("#funcname kernel failed\n")

#define CUDAKERNELSTREAMSYNC(funcname, dim_grid, dim_block, n, stream, ...) \
  funcname <<< dim_grid, dim_block, n, stream >>> (__VA_ARGS__);	\
  cutilCheckMsgAndSync ("#funcname kernel failed\n")

#define CUDAKERNELSTREAM(funcname, dim_grid, dim_block, n, stream, ...) \
  funcname <<< dim_grid, dim_block, n, stream >>> (__VA_ARGS__);	\
  cutilCheckMsg ("#funcname kernel failed\n")


#endif


