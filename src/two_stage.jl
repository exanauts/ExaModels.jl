struct TwoStageExaModel{T, M <: ExaModel{T}}
    model::M
    ns::Int
    nv::Int
    nd::Int
    nθ::Int
    nc_per_s::Int
end

function TwoStageExaModel(
    build_fn::Function, nd::Int, nv::Int, ns::Int, θ_sets;
    d_start = nothing, v_start = nothing,
    d_lvar = -Inf, d_uvar = Inf,
    v_lvar = -Inf, v_uvar = Inf,
)
    nθ = length(first(θ_sets))
    core = TwoStageExaCore()

    # Recourse variables first: indices 1:ns*nv
    scenario_tags = [i for i in 1:ns for _ in 1:nv]
    v_start_val = v_start !== nothing ? v_start : zeros(ns * nv)
    v = variable(core, ns * nv; start = v_start_val, lvar = v_lvar, uvar = v_uvar, scenario = scenario_tags)

    # Design variables: indices ns*nv+1 : ns*nv+nd
    d_start_val = d_start !== nothing ? d_start : zeros(nd)
    d = variable(core, nd; start = d_start_val, lvar = d_lvar, uvar = d_uvar, scenario = 0)

    # Parameters
    θ = parameter(core, vcat(θ_sets...))

    # User builds objectives and constraints
    build_fn(core, d, v, θ, ns, nv, nθ)

    m = ExaModel(core)
    nc = NLPModels.get_ncon(m)
    nc_per_s = ns > 0 ? nc ÷ ns : 0

    return TwoStageExaModel(m, ns, nv, nd, nθ, nc_per_s)
end

# Index accessors
recourse_var_indices(m::TwoStageExaModel, s::Int) = ((s-1)*m.nv+1):(s*m.nv)
design_var_indices(m::TwoStageExaModel) = (m.ns*m.nv+1):(m.ns*m.nv+m.nd)
cons_block_indices(m::TwoStageExaModel, s::Int) = ((s-1)*m.nc_per_s+1):(s*m.nc_per_s)

recourse_var_index(m::TwoStageExaModel, s::Int, j::Int) = (s-1)*m.nv + j
design_var_index(m::TwoStageExaModel, j::Int) = m.ns*m.nv + j
global_con_index(m::TwoStageExaModel, s::Int, j::Int) = (s-1)*m.nc_per_s + j

grad_recourse_indices(m::TwoStageExaModel, s::Int) = recourse_var_indices(m, s)
grad_design_indices(m::TwoStageExaModel) = design_var_indices(m)

# Extraction helpers
function extract_recourse_vars!(vec, m::TwoStageExaModel, s::Int, x)
    copyto!(vec, 1, x, (s-1)*m.nv+1, m.nv)
end

function extract_design_vars!(vec, m::TwoStageExaModel, x)
    copyto!(vec, 1, x, m.ns*m.nv+1, m.nd)
end

# Dimension getters
total_vars(m::TwoStageExaModel) = m.ns * m.nv + m.nd
total_cons(m::TwoStageExaModel) = NLPModels.get_ncon(m.model)

# Parameter management
function set_scenario_parameters!(m::TwoStageExaModel, s::Int, θ_vals)
    offset = (s-1) * m.nθ
    copyto!(m.model.θ, offset + 1, θ_vals, 1, m.nθ)
end

function set_all_scenario_parameters!(m::TwoStageExaModel, θ_sets)
    for (s, θ_vals) in enumerate(θ_sets)
        set_scenario_parameters!(m, s, θ_vals)
    end
end

# NLPModels forwarding
NLPModels.get_nvar(m::TwoStageExaModel) = NLPModels.get_nvar(m.model)
NLPModels.get_ncon(m::TwoStageExaModel) = NLPModels.get_ncon(m.model)
NLPModels.get_nnzj(m::TwoStageExaModel) = NLPModels.get_nnzj(m.model)
NLPModels.get_nnzh(m::TwoStageExaModel) = NLPModels.get_nnzh(m.model)
