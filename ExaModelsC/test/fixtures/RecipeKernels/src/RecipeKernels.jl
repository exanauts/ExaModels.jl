"""
    RecipeKernels

A stand-in for a modelling library, in the one respect that matters here: it
owns a function that ends up *inside* a recipe rather than beside it.

A starting point that varies with the index cannot be a number in the core, and
the size it runs over is not known while the structure is written — so it is
deferred, and the deferred thing is a function this package wrote.  The core
that comes out therefore names a `RecipeKernels` type, and a process that
cannot load `RecipeKernels` cannot read that core back.

The functions are deliberately named rather than anonymous: a `Base.Generator`
over `alternating` carries `typeof(alternating)`, which is stable across
recompiles, whereas a closure carries a gensym that is not.
"""
module RecipeKernels

export alternating, offsets

"Alternating starting point — the shape a per-index start generator takes."
@inline alternating(i) = isodd(i) ? -1.2 : 1.0

"An index set computed from the size: a comprehension, so it has no symbolic
form and has to run once the size is known."
offsets(n) = [2 * div(i - 1, 2) for i in 1:2:max(n - 2, 0)]

end # module
