.SUFFIXES: .cu .cuh .h

INC = -I. -Iinclude_cuda
LIBDIR = -L/usr/local/packages/cuda/4.2.9/cuda/lib64 -L/usr/local/packages/hdf5/1.8.10/Intel-13.0.0-openmpi-1.6.2/lib
LIB = -lcudart #-lhdf5 #-lcutil_x86_64
#-lm

CC = h5pcc
CPP = h5pcc
CXX = nvcc
MPICC = h5pcc

# -v
ARCHFLAG = -gencode arch=compute_20,code=sm_20
OPTFLAG0 = -O0 -g -G
OPTFLAG1 = -O0
OPTFLAG2 = -O2
OPTFLAG3 = -O3
ASMFLAG = -ptx
PROFFLAG = --ptxas-options=-v -keep
#WFLAGS = -Wall # not supported by nvcc

CFLAGS = -std=c99 -pthread
CPPFLAGS = #-pthread
CXXFLAGS = $(INC) $(LIB) $(LIBDIR) $(ARCHFLAG)

srcdir = src
gpusrc = gpu.cu #threads.cu
cpusrc = host_main.cpp host_func.cpp host_launcher.cu lcg.cpp
exec = mpi_ising #postprocess_ea postprocess_ea2 postprocess_q postprocess_qq postprocess_r12

default: $(exec) 


#.cu: Makefile
#	$(CXX) $(CFLAGS) -o $@ $*.cu
#


%.o: %.c
	$(CC) $(OPTFLAG2) $(CFLAGS) -c $<

%.o: %.cpp
	$(CPP) $(OPTFLAG2) $(CPPFLAGS) -c $<

%.o: %.cu
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -c $<

mpiprocess.o: mpiprocess.cpp
	$(MPICC) -c $<

mpi_ising: host_main.o mpiprocess.o threads.o host_func.o host_launcher.o lcg.o gpu.o host_kernel.o
	$(MPICC) $(OPTFLAG2) $(LIB) $(LIBDIR) -o $@ $^

ising: host_main.o threads.o host_func.o host_launcher.o lcg.o gpu.o host_kernel.o
	$(CXX) $(OPTFLAG2) $(CXXFLAGS) -o $@ $^

postprocess_%: postprocess_%.c
	$(CPP) -O2 $(CPPFLAGS) -o $@ $<



prof: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) $^

profclean: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(PROFFLAG) -clean $^

g_ising: $(gpusrc) $(cpusrc)
	$(CXX) $(OPTFLAG0) $(CXXFLAGS) -o $@ $^

#ptx: $(src)
#	$(CXX) $(OPTFLAG1) $(CXXFLAGS) $(ASMFLAG) $^



install:
	cp $(exec) /work/sfeng/ising/short_l10/

clean:
	rm -rf *.o $(exec)

cleanoutput:
	rm -rf output_*

