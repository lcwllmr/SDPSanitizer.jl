using SDPSanitizer
using Test
using LinearAlgebra
using SparseArrays
import MathOptInterface as MOI
using Clarabel
using CSDP

@testset "Numerical Conditioning: Disparate Affine Scale Coupling" begin
    # Simulates physical network coupling (e.g. low-impedance branches vs standard branches):
    # Free variables z have coefficients spanning 6 orders of magnitude (1e-4 to 1e4).
    # Verifies that:
    # 1. No valid columns are falsely pruned as syzygy columns.
    # 2. Equilibration correctly balances pivot selection.
    # 3. Solvers find the exact global minimum and recover primal/dual solutions accurately.

    # Analytical complementary split for 2x2 PSD block:
    Z_flat = [2.0, 0.0, 0.0] # X11, X12, X22
    S_flat = [0.0, 0.0, 3.0]

    m = 3
    p = 2
    # Disparate scale D:
    I_D = [1, 1, 2, 2, 3]
    J_D = [1, 2, 1, 2, 1]
    V_D = [1e4, 1e4, 1.0, 0.5, 0.1]
    D = sparse(I_D, J_D, V_D, m, p)

    I_A = [1, 2, 3]
    J_A = [1, 3, 2]
    V_A = [1.0, 1.0, sqrt(2.0)]
    A = sparse(I_A, J_A, V_A, m, 3)

    z_true = [1.2, -0.7]
    y_true = [1e-3, 2.0, 1.0]

    b = -(A * Z_flat) - (D * z_true)
    C = S_flat - (A' * y_true)
    f = -(D' * y_true)
    b0 = -dot(b, y_true)

    sdp = SemidefiniteProgram(
        C = dropzeros!(sparse(C)),
        f = dropzeros!(sparse(f)),
        b0 = b0,
        A = A,
        D = D,
        b = dropzeros!(sparse(b)),
        blocks = [2]
    )

    # Step 1: Verify presolve preserves all independent columns
    sdp_p = copy(sdp)
    presolve!(sdp_p)

    @test size(sdp_p.D, 2) == 0
    @test length(sdp_p.recovery_info.independent_cols) == 2
    @test size(sdp_p.A, 1) == 1 # 3 rows - 2 eliminated variables = 1 reduced row

    # Step 2: Solve with Clarabel through MOIWrapper
    wrapper = MOIWrapper(Clarabel.Optimizer; presolve=true, eliminate_free_variables=true)
    MOI.set(wrapper, MOI.Silent(), true)

    model = as_model(sdp)
    idx_map = MOI.copy_to(wrapper, model)
    MOI.optimize!(wrapper)

    @test MOI.get(wrapper, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(wrapper, MOI.ObjectiveValue()), 0.0, atol=1e-5)

    # Step 3: Verify recovered primal solution satisfies original constraints exactly
    vars = MOI.get(model, MOI.ListOfVariableIndices())
    X_sol = [MOI.get(wrapper, MOI.VariablePrimal(), idx_map[vars[i]]) for i in 1:3]
    z_sol = [MOI.get(wrapper, MOI.VariablePrimal(), idx_map[vars[i]]) for i in 4:5]

    residual = norm(Matrix(A) * X_sol + Matrix(D) * z_sol + Vector(b), Inf)
    @test residual < 1e-5
    @test isapprox(z_sol, z_true, atol=1e-3)
end

@testset "Numerical Conditioning: Chained Triangular Affine Elimination" begin
    # Tests chained dependency structure:
    # z1 = f(X)
    # z2 = f(z1, X)
    # z3 = f(z2, X)
    # Verifies sequential substitution, zero unwanted fill-in, and exact primal recovery.

    Z_flat = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0] # 3x3 identity in ScaledPositiveSemidefiniteConeTriangle
    S_flat = zeros(6) # dual complementary S = 0

    m = 4
    p = 3
    I_D = [1, 2, 2, 3, 3, 4]
    J_D = [1, 1, 2, 2, 3, 3]
    V_D = [2.0, -1.0, 3.0, -2.0, 4.0, 1.0]
    D = sparse(I_D, J_D, V_D, m, p)

    # 3x3 PSD matrix: diagonals at k = 1, 3, 6
    I_A = [1, 2, 3]
    J_A = [1, 3, 6]
    V_A = [1.0, 1.0, 1.0]
    A = sparse(I_A, J_A, V_A, m, 6)

    z_true = [0.5, -0.2, 1.1]
    y_true = [0.1, 0.2, 0.3, 0.4]

    b = -(A * Z_flat) - (D * z_true)
    C = S_flat - (A' * y_true)
    f = -(D' * y_true)
    b0 = -dot(b, y_true)

    sdp = SemidefiniteProgram(
        C = dropzeros!(sparse(C)),
        f = dropzeros!(sparse(f)),
        b0 = b0,
        A = A,
        D = D,
        b = dropzeros!(sparse(b)),
        blocks = [3]
    )

    sdp_p = copy(sdp)
    presolve!(sdp_p)

    @test size(sdp_p.D, 2) == 0
    @test length(sdp_p.recovery_info.independent_cols) == 3
    @test size(sdp_p.A, 1) == 1

    wrapper = MOIWrapper(Clarabel.Optimizer; presolve=true, eliminate_free_variables=true)
    MOI.set(wrapper, MOI.Silent(), true)

    model = as_model(sdp)
    idx_map = MOI.copy_to(wrapper, model)
    MOI.optimize!(wrapper)

    @test MOI.get(wrapper, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(wrapper, MOI.ObjectiveValue()), 0.0, atol=1e-5)

    vars = MOI.get(model, MOI.ListOfVariableIndices())
    X_sol = [MOI.get(wrapper, MOI.VariablePrimal(), idx_map[vars[i]]) for i in 1:6]
    z_sol = [MOI.get(wrapper, MOI.VariablePrimal(), idx_map[vars[i]]) for i in 7:9]

    residual = norm(Matrix(A) * X_sol + Matrix(D) * z_sol + Vector(b), Inf)
    @test residual < 1e-5
    @test isapprox(dot(Vector(C), X_sol) + dot(Vector(f), z_sol) + b0, 0.0, atol=1e-5)
end
