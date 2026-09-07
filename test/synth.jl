using SDPSanitizer
using Test
using LinearAlgebra
import MathOptInterface as MOI
using Clarabel
using CSDP

@testset "Correctness of pre-solver on synthetic SDPs: $label" for (label, blocks, p, m, rank_D) in [
    ("standard multi-block", [10, 5, 5], 8, 15, 4),
    ("single block", [6], 4, 10, 2),
    ("minimal 1x1 scalar block", [1], 2, 3, 1),
    ("multiple 1x1 scalar blocks (diagonal)", [1, 1, 1], 3, 4, 2),
    ("no free variables (p=0)", [4, 3], 0, 5, 0),
    ("zero-rank D (rank_D=0)", [5, 2], 3, 6, 0),
    ("single equality constraint (m=1)", [3, 2], 2, 1, 1),
    ("full-rank D (rank_D=min(m,p))", [4, 2], 3, 5, 3),
]
    # generate a random sdp with the specified characteristics
    sdp, Z, z = synthetic_sdp(blocks, p, m, rank_D)
    @test !iszero(sdp.C)
    @test p == 0 || iszero(sdp.D) || !iszero(sdp.f)
    @test isapprox(sdp.A * Z + sdp.D * z + sdp.b, zeros(m), atol=1e-5)
    @test isapprox(dot(sdp.C, Z) + dot(sdp.f, z) + sdp.b0, 0.0, atol=1e-5)

    # check that it has the specified objective value
    model = as_model(sdp)
    optimizer = MOI.instantiate(Clarabel.Optimizer, with_bridge_type=Float64)
    MOI.set(optimizer, MOI.Silent(), true)
    MOI.copy_to(optimizer, model)
    MOI.optimize!(optimizer)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    @test status == MOI.OPTIMAL
    obj_val = MOI.get(optimizer, MOI.ObjectiveValue())
    @test isapprox(obj_val, 0.0, atol=1e-5)

    # pre-solve check that we get the same optimal value
    sdp_presolved = copy(sdp)
    sdp_presolved.config.verbose = false
    presolve!(sdp_presolved)
    model = as_model(sdp_presolved)
    optimizer = MOI.instantiate(CSDP.Optimizer, with_bridge_type=Float64)
    MOI.set(optimizer, MOI.Silent(), true)
    idx_map = MOI.copy_to(optimizer, model)
    MOI.optimize!(optimizer)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    @test status == MOI.OPTIMAL
    obj_val = MOI.get(optimizer, MOI.ObjectiveValue())
    @test isapprox(obj_val, 0.0, atol=1e-5)

    # check that the recovered solutions are feasible and optimal in the original sdp
    Z_vars = MOI.get(model, MOI.ListOfVariableIndices())
    Z_sol = MOI.get(optimizer, MOI.VariablePrimal(), [idx_map[v] for v in Z_vars])
    z_sol_recovered = recover_affine_solution(sdp_presolved, Z_sol)
    @test isapprox(sdp.A * Z_sol + sdp.D * z_sol_recovered + sdp.b, zeros(m), atol=1e-5)
    @test isapprox(dot(sdp.C, Z_sol) + dot(sdp.f, z_sol_recovered) + sdp.b0, 0.0, atol=1e-5)
end
