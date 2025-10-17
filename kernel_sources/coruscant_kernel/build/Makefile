#Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)


# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT) #-G #debug flag added 
#NVCCFLAGS   += --disable-loop-unrolling
CCFLAGS     := -fPIC
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/
LIBRARIES := -lcublas 

################################################################################

# Gencode arguments
SMS ?= 89
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS +=  --threads 0 --std=c++11
ALL_CCFLAGS += -maxrregcount=255
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################

HEAD_FILES = Coruscant_SpMM_API.cuh \
			 ../csrc/Reduction_Kernel.cuh ../csrc/SpMM_Kernel.cuh \
			 ../csrc/MatMulUtilities.cuh \
			 ../csrc/MMA_PTX.cuh ../csrc/AsyncCopy_PTX.cuh \
			 ../csrc/TilingConfig.h
# Target rules
all: libCoruscant_SpMM_API.so

libCoruscant_SpMM_API.so: Coruscant_SpMM_API.o
	$(EXEC) $(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

Coruscant_SpMM_API.o:  ../csrc/Coruscant_SpMM_API.cu $(HEAD_FILES)  
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -f libCoruscant_SpMM_API.so Coruscant_SpMM_API.o
