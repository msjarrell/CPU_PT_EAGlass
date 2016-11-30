#ifndef LCG_H
#define LCG_H

// Linear congruential generator

typedef int LCG_DATATYPE;


#define LCG_A 16807
#define LCG_M 2147483647
#define LCG_Q 127773
#define LCG_R 2836


void host_init_seed (LCG_DATATYPE *gpuseed);

#endif
