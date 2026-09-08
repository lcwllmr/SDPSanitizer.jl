using SDPSanitizer
using Test
using SparseArrays
using LinearAlgebra
import MathOptInterface as MOI
using MosekTools

@testset "Large-scale SDP presolve and Mosek solve" begin
    m = 25_000
    p = 4_000

    # Blocks: 50 blocks of 15x15, 50 blocks of 20x20, and 20,000 1x1 blocks
    blocks = vcat(fill(15, 50), fill(20, 50), fill(1, 20_000))
    n_conic = sum(div.(blocks .* (blocks .+ 1), 2))

    println("\n=== Setting up large SDP ===")
    println("Constraints m = $m, Affine variables p = $p, Conic variables = $n_conic")
    println("Block structure: $(length(blocks)) blocks")

    # 1. Generate sparse constraint matrix A (~5 conic vars per constraint)
    I_A = Int[]; J_A = Int[]; V_A = Float64[]
    for i in 1:m
        for _ in 1:5
            push!(I_A, i)
            push!(J_A, rand(1:n_conic))
            push!(V_A, randn())
        end
    end
    A = sparse(I_A, J_A, V_A, m, n_conic)

    # 2. Generate sparse affine matrix D (rank p = 4,000)
    D_I = Int[]; D_J = Int[]; D_V = Float64[]
    for i in 1:p
        push!(D_I, i)
        push!(D_J, i)
        push!(D_V, randn() + 2.0)
        if i < p && rand() < 0.3
            push!(D_I, i)
            push!(D_J, i + 1)
            push!(D_V, randn())
        end
    end
    for i in (p+1):min(m, p + 500)
        for _ in 1:2
            push!(D_I, i)
            push!(D_J, rand(1:p))
            push!(D_V, randn())
        end
    end
    D = sparse(D_I, D_J, D_V, m, p)

    b = sparse(randn(m))
    C = sparse(randn(n_conic))
    f = sparse(randn(p))

    sdp = SemidefiniteProgram(
        C = C,
        f = f,
        b0 = 0.0,
        A = A,
        D = D,
        b = b,
        blocks = blocks,
        config = SanitizerConfig(verbose = true)
    )

    # 3. Test presolve! performance and correctness of dimensional reduction
    println("\n=== Running presolve! ===")
    @time presolve!(sdp)

    @test size(sdp.D, 2) == 0
    @test size(sdp.A, 1) == m - p
    @test length(sdp.b) == m - p
    @test length(sdp.C) == n_conic
    @test nnz(sdp.A) > 0

    println("\nPresolve successfully reduced constraints from $m to $(size(sdp.A, 1))")
    println("Preserved sparsity: nnz(A) = $(nnz(sdp.A))")

    # 4. Attempt solve with Mosek
    println("\n=== Translating to MOI model and solving with Mosek ===")
    model = as_model(sdp)
    optimizer = MOI.instantiate(Mosek.Optimizer, with_bridge_type=Float64)
    MOI.set(optimizer, MOI.Silent(), false)

    MOI.copy_to(optimizer, model)
    MOI.optimize!(optimizer)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    println("Mosek solve completed with status: $status")
    @test status in (
        MOI.OPTIMAL, MOI.ALMOST_OPTIMAL,
        MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED,
        MOI.SLOW_PROGRESS, MOI.ITERATION_LIMIT
    )
end
