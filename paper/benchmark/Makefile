JULIA    ?= julia
PROJECT  := --project=.
SECONDS  ?= 2.0

# SSH support: make cuda SSH_HOST=remote-host
# Runs benchmark on remote, retrieves results to local results/
SSH_HOST ?=
REMOTE_DIR ?= ExaModels.jl/paper/benchmark

ifeq ($(SSH_HOST),)
  RUN =
  FETCH_RESULTS =
else
  RUN = ssh $(SSH_HOST) "cd $(REMOTE_DIR) &&"
  FETCH_RESULTS = @mkdir -p results && scp $(SSH_HOST):$(REMOTE_DIR)/results/*.csv $(SSH_HOST):$(REMOTE_DIR)/results/*_hw.txt results/ 2>/dev/null || true
endif

# ============================================================================
# Benchmark targets — ExaModels (various backends × precisions)
# ============================================================================

# ExaModels on CPU (nothing backend, no KA), fp64
benchmark:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl nothing $(SECONDS)
	$(FETCH_RESULTS)

# ExaModels on CPU, fp32
benchmark-fp32:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl nothing $(SECONDS) fp32
	$(FETCH_RESULTS)

# ExaModels on KernelAbstractions CPU (8 threads), fp64
cpu8:
	$(RUN) $(JULIA) $(PROJECT) -t 8 benchmark.jl CPU $(SECONDS)
	$(FETCH_RESULTS)

# ExaModels on KernelAbstractions CPU (8 threads), fp32
cpu8-fp32:
	$(RUN) $(JULIA) $(PROJECT) -t 8 benchmark.jl CPU $(SECONDS) fp32
	$(FETCH_RESULTS)

# ExaModels on CUDA GPU, fp64
cuda:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl CUDA $(SECONDS)
	$(FETCH_RESULTS)

# ExaModels on CUDA GPU, fp32
cuda-fp32:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl CUDA $(SECONDS) fp32
	$(FETCH_RESULTS)

# ExaModels on AMD GPU
amdgpu:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl AMDGPU $(SECONDS)
	$(FETCH_RESULTS)

# ExaModels on Intel GPU
oneapi:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl oneAPI $(SECONDS)
	$(FETCH_RESULTS)

# ExaModels on Apple Metal GPU, fp32 (Metal does not support fp64)
metal:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl Metal $(SECONDS) fp32
	$(FETCH_RESULTS)

# ============================================================================
# Reference: JuMP + AMPL (CPU, single-threaded, fp64 only)
# ============================================================================

reference:
	$(RUN) $(JULIA) $(PROJECT) benchmark.jl reference $(SECONDS)
	$(FETCH_RESULTS)

# ============================================================================
# Post-processing targets
# ============================================================================

collect:
	$(JULIA) $(PROJECT) collect.jl

plot: collect
	$(JULIA) $(PROJECT) plot.jl

table: collect
	$(JULIA) $(PROJECT) table.jl

# ============================================================================
# Convenience
# ============================================================================

# Run everything on CPU (ExaModels fp64 + fp32 + reference)
all: benchmark benchmark-fp32 reference

clean:
	rm -rf results

.PHONY: benchmark benchmark-fp32 cpu8 cpu8-fp32 cuda cuda-fp32 amdgpu oneapi metal reference collect plot table all clean
