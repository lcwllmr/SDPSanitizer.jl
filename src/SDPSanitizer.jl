module SDPSanitizer

using LinearAlgebra
using SparseArrays
import MathOptInterface as MOI

include("sdp.jl")
export SemidefiniteProgram, SanitizerConfig, as_model

include("presolve.jl")
export presolve!, recover_affine_solution, recover_dual_solution

include("synth.jl")
export synthetic_sdp

include("wrapper.jl")
export MOIWrapper

end
