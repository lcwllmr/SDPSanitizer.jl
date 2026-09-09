using SDPSanitizer
using Test
using SparseArrays
using LinearAlgebra
using Random
import MathOptInterface as MOI
using Clarabel

function generate_reducible_large_sdp(; seed::Int = 42)
    Random.seed!(seed)

    # Block structure:
    # 5 active blocks of size 10 (5 * 55 = 275 vars)
    # 20 inactive blocks of size 10 (20 * 55 = 1100 vars)
    # 10 active scalar blocks of size 1 (10 vars)
    # 2000 inactive scalar blocks of size 1 (2000 vars)
    blocks = vcat(fill(10, 5), fill(10, 20), fill(1, 10), fill(1, 2000))
    n_blocks = length(blocks)

    block_offsets = Int[0]
    for b in blocks
        push!(block_offsets, block_offsets[end] + div(b * (b + 1), 2))
    end
    n_conic = block_offsets[end]

    function get_k(b_idx, i, j)
        @assert i <= j
        return block_offsets[b_idx] + i + div(j * (j - 1), 2)
    end

    p = 1000  # number of free variables

    I_A = Int[]; J_A = Int[]; V_A = Float64[]
    I_D = Int[]; J_D = Int[]; V_D = Float64[]
    b_vec = Float64[]

    # 1. Rows 1:p: basis rows for D
    for r in 1:p
        push!(I_D, r); push!(J_D, r); push!(V_D, 2.0)
        push!(I_A, r); push!(J_A, rand(1:n_conic)); push!(V_A, 0.1 * randn())
        push!(b_vec, 0.5 * randn())
    end

    # 2. Linear combination rows (1500 rows that are linear combinations of rows 1:p)
    # Presolve's elimination of z causes these rows to cancel out to 0 = 0,
    # which Sieve-SDP subsequently detects and deletes.
    n_lin = 1500
    for l in 1:n_lin
        r = p + l
        r1 = rand(1:p)
        r2 = rand(1:p)
        w1 = 0.7
        w2 = -0.3

        push!(I_D, r); push!(J_D, r1); push!(V_D, w1 * 2.0)
        push!(I_D, r); push!(J_D, r2); push!(V_D, w2 * 2.0)

        for ptr in 1:length(I_A)
            if I_A[ptr] == r1
                push!(I_A, r); push!(J_A, J_A[ptr]); push!(V_A, w1 * V_A[ptr])
            elseif I_A[ptr] == r2
                push!(I_A, r); push!(J_A, J_A[ptr]); push!(V_A, w2 * V_A[ptr])
            end
        end
        push!(b_vec, w1 * b_vec[r1] + w2 * b_vec[r2])
    end

    # 3. Facial reduction constraints: X_{ii} = 0 for all inactive blocks.
    # Since X ⪰ 0, X_{ii} = 0 forces the entire row and column i of the block to 0.
    curr_row = p + n_lin
    for b_idx in 6:25
        d = blocks[b_idx]
        for i in 1:d
            curr_row += 1
            k = get_k(b_idx, i, i)
            push!(I_A, curr_row); push!(J_A, k); push!(V_A, 1.0)
            push!(b_vec, 0.0)
        end
    end
    for b_idx in 36:n_blocks
        curr_row += 1
        k = get_k(b_idx, 1, 1)
        push!(I_A, curr_row); push!(J_A, k); push!(V_A, 1.0)
        push!(b_vec, 0.0)
    end

    # 4. Dummy constraints on inactive blocks that become 0 = 0 once the blocks collapse.
    n_dummy = 2000
    for _ in 1:n_dummy
        curr_row += 1
        b_idx = rand(6:25)
        d = blocks[b_idx]
        i = rand(1:d); j = rand(i:d)
        k = get_k(b_idx, i, j)
        push!(I_A, curr_row); push!(J_A, k); push!(V_A, randn())
        push!(b_vec, 0.0)
    end

    # 5. Core active constraints on active blocks:
    # Tr(X^(b)) = 1.0 for active blocks 1:5
    for b_idx in 1:5
        curr_row += 1
        d = blocks[b_idx]
        for i in 1:d
            k = get_k(b_idx, i, i)
            push!(I_A, curr_row); push!(J_A, k); push!(V_A, 1.0)
        end
        push!(b_vec, -1.0)
    end
    # x_i = 0.5 for active scalar blocks 26:35
    for b_idx in 26:35
        curr_row += 1
        k = get_k(b_idx, 1, 1)
        push!(I_A, curr_row); push!(J_A, k); push!(V_A, 1.0)
        push!(b_vec, -0.5)
    end

    m = curr_row

    A = sparse(I_A, J_A, V_A, m, n_conic)
    D = sparse(I_D, J_D, V_D, m, p)
    b = sparse(b_vec)

    # Well-scaled positive diagonal objective:
    # 1.0 on active block diagonals, 2.0 on active scalar blocks
    C_I = Int[]; C_V = Float64[]
    for b_idx in 1:5
        d = blocks[b_idx]
        for i in 1:d
            push!(C_I, get_k(b_idx, i, i)); push!(C_V, 1.0)
        end
    end
    for b_idx in 26:35
        push!(C_I, get_k(b_idx, 1, 1)); push!(C_V, 2.0)
    end
    C = sparsevec(C_I, C_V, n_conic)
    f = spzeros(p)

    return SemidefiniteProgram(
        C = C,
        f = f,
        b0 = 0.0,
        A = A,
        D = D,
        b = b,
        blocks = blocks,
        config = SanitizerConfig(verbose = false)
    )
end

@testset "Large-scale SDP presolve, sieve, and solve" begin
    sdp = generate_reducible_large_sdp()
    m_orig = size(sdp.A, 1)
    p_orig = size(sdp.D, 2)
    n_conic = size(sdp.A, 2)

    @test m_orig > 6000
    @test p_orig == 1000

    # 1. Step 1: Presolve eliminates all p free variables
    presolve!(sdp)
    @test size(sdp.D, 2) == 0
    @test size(sdp.A, 1) == m_orig - p_orig

    # 2. Step 2: Sieve-SDP identifies redundant constraints and collapses inactive blocks
    active_con = sieve!(sdp)
    # Exactly 15 core constraints remain active; 3500 redundant constraints were dropped
    @test length(active_con) == 15
    @test size(sdp.A, 1) < m_orig - p_orig

    # 3. Step 3: Solve with Clarabel in CI
    model = as_model(sdp)
    optimizer = MOI.instantiate(Clarabel.Optimizer, with_bridge_type=Float64)
    MOI.set(optimizer, MOI.Silent(), true)
    MOI.copy_to(optimizer, model)
    MOI.optimize!(optimizer)

    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    obj_val = MOI.get(optimizer, MOI.ObjectiveValue())
    # Expected objective: 5 * 1.0 + 10 * 0.5 * 2.0 = 15.0
    @test isapprox(obj_val, 15.0, atol=1e-4)

    # 4. Step 4: Full unreduced pipeline through MOIWrapper
    orig_sdp = generate_reducible_large_sdp()
    orig_model = as_model(orig_sdp)
    wrapper_opt = MOIWrapper(Clarabel.Optimizer, presolve=true, sieve=true)
    MOI.set(wrapper_opt, MOI.Silent(), true)
    MOI.copy_to(wrapper_opt, orig_model)
    MOI.optimize!(wrapper_opt)

    @test MOI.get(wrapper_opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(wrapper_opt, MOI.ObjectiveValue()), 15.0, atol=1e-4)
end
