#ifndef HOST_KERNEL_H
#define HOST_KERNEL_H

//void host_kernel (MSC_DATATYPE * lattice, Temp * temp, double *e, int mod);

void host_kernel_warmup (MSC_DATATYPE * lattice, Temp * temp);
void host_kernel_swap (MSC_DATATYPE * lattice, Temp * temp, St * st, int rec);

#endif
