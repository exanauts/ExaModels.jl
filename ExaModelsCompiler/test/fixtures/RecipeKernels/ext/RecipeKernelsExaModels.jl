module RecipeKernelsExaModels

import ExaModels
import RecipeKernels

"A start kernel owned by the EXTENSION rather than by the package, so a core
built with it names a module that cannot be a dependency of anything."
@inline ramp(i) = 0.25 * i

"""A tuple-returning argument function owned by the EXTENSION, for the argfun
guard test: it must get past the tuple-return check so the guard itself is what
refuses it."""
exttables(n) = (Int(n),)

end # module
