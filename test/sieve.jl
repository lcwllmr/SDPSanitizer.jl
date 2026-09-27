using SDPSanitizer
using Test
using LinearAlgebra
using SparseArrays
import MathOptInterface as MOI
using Clarabel

@testset "Sieve-SDP tests" begin
    # 1. Simple reduction test
    # min X_11 + X_22
    # s.t. X_11 = 0
    #      X \succeq 0
    # X_11 = 0 implies row/col 1 is 0.
    
    C = sparsevec([1, 3], [1.0, 1.0], 3)
    f = spzeros(0)
    b0 = 0.0
    A = sparse([1], [1], [1.0], 1, 3)
    D = spzeros(1, 0)
    b = sparse([0.0])
    
    sdp = SemidefiniteProgram(C=C, f=f, b0=b0, A=A, D=D, b=b, blocks=[2])
    
    sieve!(sdp)
    # The constraint X_11 = 0 should be deleted, but X_11 = 0 added as new constraint
    @test size(sdp.A, 1) == 1
    @test sdp.A[1, 1] == 1.0
    @test length(sdp.b) == 1
    @test sdp.b[1] == 0.0
    
    # 2. Infeasibility test
    # X_11 + X_22 = -1, X \succeq 0
    A2 = sparse([1, 1], [1, 3], [1.0, 1.0], 1, 3)
    b2 = sparse([1.0])
    sdp2 = SemidefiniteProgram(C=C, f=f, b0=b0, A=A2, D=D, b=b2, blocks=[2])
    @test_throws ErrorException("INFEASIBLE") sieve!(sdp2)
    
    # 3. Wrapper test with MOI: Facial reduction intercepts duals
    optimizer = MOIWrapper(Clarabel.Optimizer, presolve=true, facial_reduction=true)
    MOI.set(optimizer, MOI.Silent(), true)
    
    x1, _ = MOI.add_constrained_variables(optimizer, MOI.ScaledPositiveSemidefiniteConeTriangle(2))
    
    # X_11 = 0
    func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1[1])], 0.0)
    c1 = MOI.add_constraint(optimizer, func, MOI.EqualTo(0.0))
    
    # min X_22
    obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1[3])], 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 0.0, atol=1e-5)
    
    # Dual status and constraint dual interception when facial reduction applied
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
    @test_throws ErrorException MOI.get(optimizer, MOI.ConstraintDual(), c1)

    # 4. Wrapper test with MOI: Pure detect_infeasibility (safe mode)
    # Feasible problem allows extracting duals normally
    optimizer_safe = MOIWrapper(Clarabel.Optimizer, presolve=true, detect_infeasibility=true, facial_reduction=false)
    MOI.set(optimizer_safe, MOI.Silent(), true)
    x_s, _ = MOI.add_constrained_variables(optimizer_safe, MOI.ScaledPositiveSemidefiniteConeTriangle(2))
    func_s = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x_s[1]), MOI.ScalarAffineTerm(1.0, x_s[3])], 0.0)
    c_s = MOI.add_constraint(optimizer_safe, func_s, MOI.EqualTo(1.0))
    obj_s = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x_s[1])], 0.0)
    MOI.set(optimizer_safe, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_s)
    MOI.set(optimizer_safe, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(optimizer_safe)
    @test MOI.get(optimizer_safe, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer_safe, MOI.DualStatus()) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    dual_val = MOI.get(optimizer_safe, MOI.ConstraintDual(), c_s)
    @test isfinite(dual_val)
    
    # 5. Wrapper test with MOI: Infeasible detection
    optimizer2 = MOIWrapper(Clarabel.Optimizer, presolve=true, detect_infeasibility=true)
    MOI.set(optimizer2, MOI.Silent(), true)
    x2, _ = MOI.add_constrained_variables(optimizer2, MOI.ScaledPositiveSemidefiniteConeTriangle(2))
    func2 = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x2[1])], 0.0)
    MOI.add_constraint(optimizer2, func2, MOI.EqualTo(-1.0))
    MOI.optimize!(optimizer2)
    @test MOI.get(optimizer2, MOI.TerminationStatus()) == MOI.INFEASIBLE
end
