using SDPSanitizer
using Test
using LinearAlgebra
using SparseArrays
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

@testset "Correctness of redundant constraint elimination in presolve" begin
    blocks = [5, 4]
    p = 3
    m_base = 6
    rank_D = 2
    sdp_base, Z_true, z_true = synthetic_sdp(blocks, p, m_base, rank_D)

    # Append 3 redundant constraints as linear combinations of the base constraints
    comb = [
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, -1.0, 0.0, 0.0],
        [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    A_red = vcat([sum(comb[k][i] * sdp_base.A[i:i, :] for i in 1:m_base) for k in 1:3]...)
    D_red = vcat([sum(comb[k][i] * sdp_base.D[i:i, :] for i in 1:m_base) for k in 1:3]...)
    b_red = [sum(comb[k][i] * sdp_base.b[i] for i in 1:m_base) for k in 1:3]

    sdp_with_redundancy = copy(sdp_base)
    sdp_with_redundancy.A = vcat(sdp_base.A, A_red)
    sdp_with_redundancy.D = vcat(sdp_base.D, D_red)
    sdp_with_redundancy.b = sparse(vcat(Vector(sdp_base.b), b_red))

    m_total = size(sdp_with_redundancy.A, 1)
    @test m_total == 9

    # 1. Presolve with eliminate_redundant_constraints = false (free vars only)
    sdp_no_red = copy(sdp_with_redundancy)
    sdp_no_red.config.eliminate_free_variables = true
    sdp_no_red.config.eliminate_redundant_constraints = false
    presolve!(sdp_no_red)
    # Since rank(D) = 2, 2 rows used to eliminate free vars, leaving 9 - 2 = 7 rows
    @test size(sdp_no_red.A, 1) == 7

    # 2. Presolve with eliminate_redundant_constraints = true
    sdp_full = copy(sdp_with_redundancy)
    sdp_full.config.eliminate_free_variables = true
    sdp_full.config.eliminate_redundant_constraints = true
    presolve!(sdp_full)
    # The 3 linear combination rows are eliminated, leaving 6 - 2 = 4 independent rows!
    @test size(sdp_full.A, 1) == 4

    # Solve the reduced problem with CSDP
    model = as_model(sdp_full)
    opt = MOI.instantiate(CSDP.Optimizer, with_bridge_type=Float64)
    MOI.set(opt, MOI.Silent(), true)
    idx_map = MOI.copy_to(opt, model)
    MOI.optimize!(opt)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(opt, MOI.ObjectiveValue()), 0.0, atol=1e-5)

    # Recover primal solution
    Z_vars = MOI.get(model, MOI.ListOfVariableIndices())
    Z_sol = MOI.get(opt, MOI.VariablePrimal(), [idx_map[v] for v in Z_vars])
    z_sol = recover_affine_solution(sdp_full, Z_sol)
    @test isapprox(sdp_with_redundancy.A * Z_sol + sdp_with_redundancy.D * z_sol + sdp_with_redundancy.b, zeros(m_total), atol=1e-5)

    # Recover dual solution
    all_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}())
    y_presolved = [MOI.get(opt, MOI.ConstraintDual(), c) for c in all_cons]
    y_recovered = recover_dual_solution(sdp_full, y_presolved)
    @test length(y_recovered) == m_total
    @test isapprox(dot(sdp_with_redundancy.C, Z_sol) + dot(sdp_with_redundancy.f, z_sol) + sdp_with_redundancy.b0, 0.0, atol=1e-5)

    # 3. Test infeasibility detection on inconsistent redundant constraint
    sdp_infeas = copy(sdp_with_redundancy)
    sdp_infeas.b[end] += 10.0
    sdp_infeas.config.eliminate_redundant_constraints = true
    @test_throws ErrorException("INFEASIBLE") presolve!(sdp_infeas)
end
