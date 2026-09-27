module SDPSanitizer

using LinearAlgebra
using SparseArrays
import MathOptInterface as MOI

include("sdp.jl")
export SemidefiniteProgram, SanitizerConfig, as_model

include("presolve.jl")
export presolve!, recover_affine_solution, recover_dual_solution

include("sieve.jl")
export sieve!

include("synth.jl")
export synthetic_sdp

include("wrapper.jl")
export MOIWrapper, get_last_inner_solve_time, get_last_total_solve_time

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    redirect_stdout(devnull) do
        @compile_workload begin
            sdp, Z, _ = synthetic_sdp([2, 2], 2, 4, 2)
            sdp.blocks = [4]
            sdp.config.verbose = true
            sdp.config.eliminate_redundant_constraints = true
            model = as_model(sdp)::MOI.Utilities.Model{Float64}
            sieve!(copy(sdp); only_detect_infeasibility=true)
            sieve!(copy(sdp); only_detect_infeasibility=false)

            psd_v = MOI.add_variables(model, 3)
            MOI.add_constraint(model, MOI.VectorOfVariables(psd_v), MOI.PositiveSemidefiniteConeTriangle(2))
            gt_v = MOI.add_variable(model)
            MOI.add_constraint(model, gt_v, MOI.GreaterThan(0.0))
            vaf = MOI.VectorAffineFunction([MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, psd_v[1])), MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, gt_v))], [0.0])
            MOI.add_constraint(model, vaf, MOI.Zeros(1))

            mock = MOI.Utilities.MockOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()))
            wrapper = MOIWrapper(mock; presolve=true, eliminate_free_variables=true, eliminate_redundant_constraints=true, detect_infeasibility=true, verbose=true)
            MOI.copy_to(wrapper, model)
            MOI.optimize!(wrapper)

            presolve!(sdp)
            recover_affine_solution(sdp, Z)
            recover_dual_solution(sdp, zeros(size(sdp.A, 1)))
        end
    end
end

end
