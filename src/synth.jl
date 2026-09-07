function get_block_triu_indices(blocks::Vector{Int})
    active_indices = Int[]
    offset = 0
    for b in blocks
        for j in 1:b
            for i in 1:j
                gi = i + offset
                gj = j + offset
                k = gi + div(gj * (gj - 1), 2)
                push!(active_indices, k)
            end
        end
        offset += b
    end
    return active_indices
end

function generate_orthogonal_split(size::Int)
    M = randn(size, size)
    Q = Matrix(qr(M).Q)
    rank_Z = rand(0:size)

    eig_Z = rand(size) .* 9.9 .+ 0.1
    eig_Z[rank_Z+1:end] .= 0.0

    eig_S = rand(size) .* 9.9 .+ 0.1
    eig_S[1:rank_Z] .= 0.0

    Z = Symmetric(Q * Diagonal(eig_Z) * Q')
    S = Symmetric(Q * Diagonal(eig_S) * Q')
    return Z, S
end

"""
    synthetic_sdp(blocks::Vector{Int}, p::Int, m::Int, rank_D::Int)

Generate a solvable synthetic semidefinite program with known optimal solution `(Z_flat, z)`
and optimal objective value `0.0`.

The problem data satisfies:
- Complementary slackness: `⟨S, Z⟩ = 0` with `Z ⪰ 0, S ⪰ 0`.
- Left-hand-side constraint: `b = -(A * Z_flat) - (D * z)`, ensuring `A(Z) + D z + b = 0`.
- Dual feasibility: `C_flat = S_flat - Aᵀ y` and `f = -Dᵀ y`.
- Objective offset: `b₀ = -bᵀ y`, so `⟨C, Z⟩ + fᵀ z + b₀ = 0`.

Returns `(sdp::SemidefiniteProgram, Z_flat::Vector{Float64}, z::Vector{Float64})`.
"""
function synthetic_sdp(blocks::Vector{Int}, p::Int, m::Int, rank_D::Int)
    n = sum(blocks)
    N_triu = div(n * (n + 1), 2)

    # 1. Generate complementary Z and S
    Z = zeros(n, n)
    S = zeros(n, n)
    offset = 0
    for b in blocks
        Zb, Sb = generate_orthogonal_split(b)
        Z[offset+1:offset+b, offset+1:offset+b] .= Zb
        S[offset+1:offset+b, offset+1:offset+b] .= Sb
        offset += b
    end

    # 2. Extract scaled flattened variables
    # For Scaled PSD Cone, off-diagonals are scaled by √2 for BOTH primal and dual
    Z_flat = zeros(N_triu)
    S_flat = zeros(N_triu)
    for j in 1:n
        for i in 1:j
            k = i + div(j * (j - 1), 2)
            scale = (i == j) ? 1.0 : sqrt(2.0)
            Z_flat[k] = Z[i, j] * scale
            S_flat[k] = S[i, j] * scale
        end
    end

    # 3. Generate structured sparse A and sparse D
    active_idx = get_block_triu_indices(blocks)
    n_active = length(active_idx)

    I_A = repeat(1:m, inner=n_active)
    J_A = repeat(active_idx, outer=m)
    V_A = randn(m * n_active)
    A = sparse(I_A, J_A, V_A, m, N_triu)

    D = sparse(randn(m, rank_D) * randn(rank_D, p))

    # 4. Generate arbitrary free variables
    z = randn(p)
    y = randn(m)

    # 5. Compute problem data exactly via Euclidean inner products
    b = -(A * Z_flat) - (D * z)
    C_flat = S_flat - (A' * y)
    f = -(D' * y)
    b0 = -dot(b, y)

    sdp = SemidefiniteProgram(
        C = dropzeros!(sparse(C_flat)),
        f = dropzeros!(sparse(f)),
        b0 = b0,
        A = A,
        D = D,
        b = dropzeros!(sparse(b))
    )

    return sdp, Z_flat, z # Keep true variables for verification
end
