CUDA?=/usr/local/cuda
CXX=g++ -DLINUX
LIB=lib
ifeq ($(shell uname), Linux)
  ifeq ($(shell uname -m), x86_64)
	LIB=lib64
  endif
endif
PWD?=$(shell pwd)

INCLUDES +=\
	-I$(CUDA)/include  \
	-I/usr/include

LIBRARIES +=\
	-L$(CUDA)/$(LIB) -lcudart

NVCC=$(CUDA)/bin/nvcc
NVCCFLAGS += -ccbin $(CXX) \
	-Xcompiler -fPIC

all: cudaDeviceProp

cudaDeviceProp: cudaDeviceProp.cu
	$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@

.PHONY: clean
clean:
	@-rm cudaDeviceProp
