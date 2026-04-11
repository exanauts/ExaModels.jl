JULIA ?= julia

.PHONY: all nothing cuda amdgpu oneapi metal full compare clean \
        benchmark-main benchmark-current

# Default: CPU only
all: nothing

nothing:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main nothing
	$(JULIA) runbenchmark.jl current nothing
	$(JULIA) --project=. compare.jl

cuda:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main cuda
	$(JULIA) runbenchmark.jl current cuda
	$(JULIA) --project=. compare.jl

amdgpu:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main amdgpu
	$(JULIA) runbenchmark.jl current amdgpu
	$(JULIA) --project=. compare.jl

oneapi:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main oneapi
	$(JULIA) runbenchmark.jl current oneapi
	$(JULIA) --project=. compare.jl

metal:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main metal
	$(JULIA) runbenchmark.jl current metal
	$(JULIA) --project=. compare.jl

compare: benchmark-results-main.jld2 benchmark-results-current.jld2
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=. compare.jl

clean:
	rm -f benchmark-results-main.jld2 benchmark-results-current.jld2
