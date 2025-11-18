module ExaModelsPOCL

import ExaModels: ExaModels, NLPModels
import KernelAbstractions: POCL, POCLBackend

ExaModels.convert_array(v, backend::POCLBackend) = Array(v)

end
