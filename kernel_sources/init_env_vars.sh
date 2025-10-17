#Code refined from FlashLLM (https://github.com/AlibabaResearch/flash-llm)


# set up home path for each kernel
export FlashLLM_HOME=$(pwd)/flash_llm
export Coruscant_HOME=$(pwd)/coruscant_kernel
export Coruscant_STC_HOME=$(pwd)/coruscant_stc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FlashLLM_HOME/build:$Coruscant_HOME/build:$Coruscant_STC_HOME/build