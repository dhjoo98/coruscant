#Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)


# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS) #-G
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/ 
LIBRARIES := -lcublas 
#

ifneq ($(FlashLLM_HOME),)
INCLUDES  += -I${FlashLLM_HOME}/build
LIBRARIES += -L${FlashLLM_HOME}/build -lFlashLLM_SpMM_API
endif
ifneq ($(Coruscant_HOME),)
INCLUDES  += -I${Coruscant_HOME}/build
LIBRARIES += -L${Coruscant_HOME}/build -lCoruscant_SpMM_API
endif
ifneq ($(Coruscant_STC_HOME),)
INCLUDES  += -I${Coruscant_STC_HOME}/build
LIBRARIES += -L${Coruscant_STC_HOME}/build -lCoruscant_STC_SpMM_API
endif
################################################################################

# Gencode arguments
SMS ?= 89
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: spmm_perf 

spmm_perf.o: spmm_perf.cu spmm_test_utils.h 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

spmm_perf: spmm_perf.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f spmm_perf spmm_perf.o
