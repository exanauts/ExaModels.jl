JULIA ?= julia

.PHONY: all nothing cuda amdgpu oneapi metal full compare clean \
        benchmark-main benchmark-current

# Default: CPU only
all: nothing

nothing: clean
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main nothing
	$(JULIA) runbenchmark.jl current nothing
	$(JULIA) --project=. compare.jl

cuda: clean
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main cuda
	$(JULIA) runbenchmark.jl current cuda
	$(JULIA) --project=. compare.jl

amdgpu: clean
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main amdgpu
	$(JULIA) runbenchmark.jl current amdgpu
	$(JULIA) --project=. compare.jl

oneapi: clean
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main oneapi
	$(JULIA) runbenchmark.jl current oneapi
	$(JULIA) --project=. compare.jl

metal: clean
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) runbenchmark.jl main metal
	$(JULIA) runbenchmark.jl current metal
	$(JULIA) --project=. compare.jl

compare:
	$(JULIA) --project=. -e 'using Pkg; Pkg.update(); Pkg.instantiate()'
	$(JULIA) --project=. compare.jl

clean:
	rm -f benchmark-results-*.csv benchmark-results-*.jld2
