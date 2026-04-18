abstract type AbstractTag end
abstract type AbstractExaModelTag end
@inline append_var_tags(::Nothing, backend, len) = nothing
@inline append_con_tags(::Nothing, backend, len) = nothing
