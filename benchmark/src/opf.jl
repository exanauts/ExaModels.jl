############################################
# Coypright notice for jump_ac_power_model #
############################################
# 
# Copyright © 2022, Triad National Security, LLC All rights reserved.

# This software was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# This program is open source under the BSD-3 License.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#     Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#     Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#     Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function exa_ac_power_model(filename = "pglib_opf_case3_lmbd.m"; kwargs...)
    return ADBenchmarkModel(
        ExaModelsExamples.ac_power_model(filename; kwargs...)
    )
end
function jump_ac_power_model(filename = "pglib_opf_case3_lmbd.m")
    
    ref = ExaModelsExamples.get_power_data_ref(filename)

    model = JuMP.Model()
    #JuMP.set_optimizer_attribute(model, "print_level", 0)

    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)

    JuMP.@variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    JuMP.@variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    JuMP.@variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])

    JuMP.@objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i,bus) in ref[:ref_buses]
        JuMP.@constraint(model, va[i] == 0)
    end

    for (i,bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        JuMP.@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) -
            sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2
        )

        JuMP.@constraint(model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) -
            sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2
        )
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = ExaModelsExamples.PowerModels.calc_branch_y(branch)
        tr, ti = ExaModelsExamples.PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]

        # From side of the branch flow
        JuMP.@NLconstraint(model, p_fr ==  (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )
        JuMP.@NLconstraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)) )

        # To side of the branch flow
        JuMP.@NLconstraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )
        JuMP.@NLconstraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)) )

        # Voltage angle difference limit
        JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])

        # Apparent power limit, from side and to side
        JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    return ADBenchmarkModel(
        MathOptNLPModel(model)
    )
end
 

function ampl_data(filename)
    data = ExaModelsExamples.parse_ac_power_data(filename)
    
    bus_gens = [Int[] for i=1:length(data.bus)]
    bus_arcs = [Int[] for i=1:length(data.bus)]
    
    for g in data.gen
        push!(bus_gens[g.bus], g.i)
    end
    
    for a in data.arc
        push!(bus_arcs[a.bus], a.i)
    end
    
    return data, bus_gens, bus_arcs
end

function ampl_ac_power_model(filename = "pglib_opf_case3_lmbd.m")

    short = splitext(basename(filename))[1]
    
    nlfile = joinpath(TMPDIR , "ac_power_model_$short.nl")

    if !isfile(nlfile)
        py"""
        import pyomo.environ as pyo
        import numpy as np
        import math
        import julia
        from julia import ExaModelsBenchmark

        data, bus_gens, bus_arcs = ExaModelsBenchmark.ampl_data($filename)
        """
        
        @info "Writing nlfile for $filename"
        py"""

        nbus = len(data.bus)
        ngen = len(data.gen)
        narc = len(data.arc)

        m = pyo.ConcreteModel()

        m.va = pyo.Var(range(nbus))

        m.vm = pyo.Var(
            range(nbus),
            initialize = np.ones(nbus),
            bounds = lambda m,i: (data.vmin[i], data.vmax[i])
        )

        m.pg = pyo.Var(
            range(ngen),
            bounds = lambda m,i: (data.pmin[i], data.pmax[i])
        )

        m.qg = pyo.Var(
            range(ngen),
            bounds = lambda m,i: (data.qmin[i], data.qmax[i])
        )

        m.p = pyo.Var(
            range(narc),
            bounds = lambda m,i: (-data.rate_a[i], data.rate_a[i])
        )

        m.q = pyo.Var(
            range(narc),
            bounds = lambda m,i: (-data.rate_a[i], data.rate_a[i])
        )

        m.obj = pyo.Objective(
            expr = sum(g.cost1 * m.pg[g.i-1]**2 + g.cost2 * m.pg[g.i-1] + g.cost3 for g in data.gen),
            sense=pyo.minimize
        )

        m.c1 = pyo.ConstraintList()
        m.c2 = pyo.ConstraintList()
        m.c3 = pyo.ConstraintList()
        m.c4 = pyo.ConstraintList()
        m.c5 = pyo.ConstraintList()
        m.c6 = pyo.ConstraintList()
        m.c7 = pyo.ConstraintList()
        m.c8 = pyo.ConstraintList()
        m.c9 = pyo.ConstraintList()
        m.c10= pyo.ConstraintList()

        for i in data.ref_buses:
            m.c1.add(expr=m.va[i-1] == 0)

        for (b, amin, amax) in zip(data.branch, data.angmin, data.angmax):
            m.c2.add(
                expr =
                m.p[b.f_idx-1]
                - b.c5*m.vm[b.f_bus-1]**2
                - b.c3*(m.vm[b.f_bus-1]*m.vm[b.t_bus-1]*pyo.cos(m.va[b.f_bus-1]-m.va[b.t_bus-1]))
                - b.c4*(m.vm[b.f_bus-1]*m.vm[b.t_bus-1]*pyo.sin(m.va[b.f_bus-1]-m.va[b.t_bus-1]))
                == 0
            )
            m.c3.add(
                expr = 
                m.q[b.f_idx-1]
                + b.c6*m.vm[b.f_bus-1]**2
                + b.c4*(m.vm[b.f_bus-1]*m.vm[b.t_bus-1]*pyo.cos(m.va[b.f_bus-1]-m.va[b.t_bus-1]))
                - b.c3*(m.vm[b.f_bus-1]*m.vm[b.t_bus-1]*pyo.sin(m.va[b.f_bus-1]-m.va[b.t_bus-1]))
                == 0
            )
            m.c4.add(
                m.p[b.t_idx-1]
                - b.c7*m.vm[b.t_bus-1]**2
                - b.c1*(m.vm[b.t_bus-1]*m.vm[b.f_bus-1]*pyo.cos(m.va[b.t_bus-1]-m.va[b.f_bus-1]))
                - b.c2*(m.vm[b.t_bus-1]*m.vm[b.f_bus-1]*pyo.sin(m.va[b.t_bus-1]-m.va[b.f_bus-1]))
                == 0
            )
            m.c5.add(
                m.q[b.t_idx-1]
                + b.c8*m.vm[b.t_bus-1]**2 
                + b.c2*(m.vm[b.t_bus-1]*m.vm[b.f_bus-1]*pyo.cos(m.va[b.t_bus-1]-m.va[b.f_bus-1]))
                - b.c1*(m.vm[b.t_bus-1]*m.vm[b.f_bus-1]*pyo.sin(m.va[b.t_bus-1]-m.va[b.f_bus-1]))
                == 0
            )
            m.c6.add(
                (amin, m.va[b.f_bus-1] - m.va[b.t_bus-1], amax)
            )
            m.c7.add(
                (None, m.p[b.f_idx-1]**2 + m.q[b.f_idx-1]**2 - b.rate_a_sq, 0)
            )
            m.c8.add(
                (None, m.p[b.t_idx-1]**2 + m.q[b.t_idx-1]**2 - b.rate_a_sq, 0)
            )

        for (b,g,a) in zip(data.bus, bus_gens, bus_arcs):
            m.c9.add(
                b.pd
                + sum(m.p[j-1] for j in a)
                - sum(m.pg[j-1] for j in g)
                + b.gs * m.vm[b.i-1]**2
                == 0
            )
            m.c10.add(
                b.qd
                + sum(m.q[j-1] for j in a)
                - sum(m.qg[j-1] for j in g)
                - b.bs * m.vm[b.i-1]**2
                == 0
            )

        m.write($nlfile)
        """
    end
    
    return ADBenchmarkModel(
        AmplNLReader.AmplModel(nlfile)
    )
end

