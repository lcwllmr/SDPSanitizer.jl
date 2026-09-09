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
    
    # 3. Wrapper test with MOI
    # Construct model using MOI directly
    optimizer = MOIWrapper(Clarabel.Optimizer, presolve=true, sieve=true)
    MOI.set(optimizer, MOI.Silent(), true)
    
    x1, _ = MOI.add_constrained_variables(optimizer, MOI.ScaledPositiveSemidefiniteConeTriangle(2))
    
    # X_11 = 0
    func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1[1])], 0.0)
    MOI.add_constraint(optimizer, func, MOI.EqualTo(0.0))
    
    # min X_22
    obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x1[3])], 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(optimizer, MOI.ObjectiveValue()), 0.0, atol=1e-5)
    
    # Dual zero-padding
    @test isapprox(MOI.get(optimizer, MOI.ConstraintDual(), MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(1)), 0.0, atol=1e-5)
    
    # Wrapper test with MOI: Infeasible
    optimizer2 = MOIWrapper(Clarabel.Optimizer, presolve=true, sieve=true)
    MOI.set(optimizer2, MOI.Silent(), true)
    x2, _ = MOI.add_constrained_variables(optimizer2, MOI.ScaledPositiveSemidefiniteConeTriangle(2))
    func2 = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x2[1])], 0.0)
    MOI.add_constraint(optimizer2, func2, MOI.EqualTo(-1.0))
    MOI.optimize!(optimizer2)
    @test MOI.get(optimizer2, MOI.TerminationStatus()) == MOI.INFEASIBLE
end
