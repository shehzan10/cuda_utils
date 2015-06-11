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
	-L$(CUDA)/$(LIB) -lcuda -lcudart -lnvToolsExt

NVCC=$(CUDA)/bin/nvcc
CUDA_OPTIMISE=-O3
NVCCFLAGS += -ccbin $(CXX) $(ARCH_FLAGS) $(CUDA_DEBUG) $(CUDA_OPTIMISE)\
	--ptxas-options=-v \
	-Xcompiler -fPIC

ifeq ($(shell uname -m), armv7l) # Tegra TK1
	NVCCFLAGS += -gencode=arch=compute_32,code=sm_32
else
	NVCCFLAGS += -gencode=arch=compute_30,code=sm_30
endif


all: bin/cudaBenchmark bin/transpose bin/transpose0

bin/cudaBenchmark: src/cudaBenchmark.cu
	$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@

bin/transpose: src/transpose.cu
	$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@

bin/transpose0: src/transpose_zero_copy.cu
	$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@

.PHONY: clean
clean:
	@echo "Cleaning..."
	@-rm bin/transpose
	@-rm bin/transpose0
	@-rm bin/cudaBenchmark
	@echo "Complete!"
