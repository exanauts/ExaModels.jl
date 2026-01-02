# Developing Extensions

ExaModels.jl's API only uses simple julia functions, and thus, implementing the extensions is straightforward. Below, we suggest a good practice for implementing an extension package.

Let's say that we want to implement an extension package for the example problem in [Getting Started](@ref guide). An extension package may look like:
```
Root
├───Project.toml
├── src
│   └── LuksanVlcekModels.jl
└── test
    └── runtest.jl
```
Each of the files containing
```toml
# Project.toml

name = "LuksanVlcekModels"
uuid = "0c5951a0-f777-487f-ad29-fac2b9a21bf1"
authors = ["Sungho Shin <sshin@anl.gov>"]
version = "0.1.0"

[deps]
ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"

[extras]
NLPModelsIpopt = "f4238b75-b362-5c4c-b852-0801c9a21d71"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test", "NLPModelsIpopt"]
```

```julia
# src/LuksanVlcekModels.jl

module LuksanVlcekModels

import ExaModels

function luksan_vlcek_obj(x,i)
    return 100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2
end

function luksan_vlcek_con(x,i)
    return 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
end

function luksan_vlcek_x0(i)
    return mod(i,2)==1 ? -1.2 : 1.0
end

function luksan_vlcek_model(N; backend = nothing)
    
    c = ExaModels.ExaCore(backend)
    x = ExaModels.variable(
        c, N;
        start = (luksan_vlcek_x0(i) for i=1:N)
    )
    ExaModels.constraint(
        c,
        luksan_vlcek_con(x,i)
        for i in 1:N-2)
    ExaModels.objective(c, luksan_vlcek_obj(x,i) for i in 2:N)
    
    return ExaModels.ExaModel(c) # returns the model
end

export luksan_vlcek_model

end # module LuksanVlcekModels
```

```julia
# test/runtest.jl

using Test, LuksanVlcekModels, NLPModelsIpopt

@testset "LuksanVlcekModelsTest" begin
    m = luksan_vlcek_model(10)
    result = ipopt(m)

    @test result.status == :first_order
    @test result.solution ≈ [
        -0.9505563573613093
        0.9139008176388945
        0.9890905176644905
        0.9985592422681151
        0.9998087408802769
        0.9999745932450963
        0.9999966246997642
        0.9999995512524277
        0.999999944919307
        0.999999930070643
    ]
    @test result.multipliers ≈ [
        4.1358568305002255
        -1.876494903703342
        -0.06556333356358675
        -0.021931863018312875
        -0.0019537261317119302
        -0.00032910445671233547
        -3.8788212776372465e-5
        -7.376592164341867e-6
    ]
end
```
