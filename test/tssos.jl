using Test
import SDPSanitizer

# derive SDP from CS-TSSOS hierarchy
using DynamicPolynomials
using JuMP
using TSSOS

# test SDPSanitizer.MOIWrapper with the following SDP solvers
import CSDP
import Clarabel
import SCS
import SDPA

@testset "TSSOS + pre-solve wrapper + $solver: Constrained O(2) spin chain with chirality" for solver in [
    CSDP.Optimizer,
    Clarabel.Optimizer,
    SCS.Optimizer,
    SDPA.Optimizer,
]
    @polyvar u1 v1 u2 v2 u3 v3 u4 v4 u5 v5
    x = [u1, v1, u2, v2, u3, v3, u4, v4, u5, v5]
    u = [1, u1, u2, u3, u4, u5, 0]
    v = [0, v1, v2, v3, v4, v5, 1]

    objective = 0
    # discrete Dirichlet energy on the unit circle
    for i in 1:6
        objective += (u[i+1] - u[i])^2 + (v[i+1] - v[i])^2
    end
    # external magentic field
    objective -= (sqrt(2) / 2) * (u[4] + v[4])

    inequalities = Poly{Float64}[]
    # global ball constraint
    push!(inequalities, 6 - sum(xi^2 for xi in x))
    # chirality / positive orientation
    for i in 1:6
        push!(inequalities, u[i] * v[i+1] - u[i+1] * v[i])
    end

    equalities = Poly{Float64}[]
    # unit circle manifold
    for i in 2:6
        push!(equalities, 1 - (u[i]^2 + v[i]^2))
    end

    pop = Poly{Float64}[]
    push!(pop, objective)
    append!(pop, inequalities)
    append!(pop, equalities)

    # Analytical global optimizer and optimum
    x_opt = [(sqrt(6)+sqrt(2))/4, (sqrt(6)-sqrt(2))/4,
             sqrt(3)/2, 1/2,
             sqrt(2)/2, sqrt(2)/2,
             1/2, sqrt(3)/2,
             (sqrt(6)-sqrt(2))/4, (sqrt(6)+sqrt(2))/4]
    fx_opt = 11 - 3 * (sqrt(6) + sqrt(2))

    model = Model(() -> SDPSanitizer.MOIWrapper(solver; presolve=true, eliminate_free_variables=true, eliminate_redundant_constraints=true, sieve=true, verbose=true))
    opt_p, sol_p, data_p = cs_tssos(pop, x, 1; numeq=length(equalities), TS="block", CS="MF", QUIET=false, solution=true, solution_mode="moment", model=model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(opt_p, fx_opt, atol=1e-4)
    @test length(sol_p) == 1
    @test isapprox(sol_p[1], x_opt, atol=1e-2)
end
