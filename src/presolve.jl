macro time_if(cond, expr)
    quote
        if $(esc(cond))
            @time $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

"""
    presolve!(sdp::SemidefiniteProgram)

Presolve `sdp` in-place by eliminating free affine variables `z`.

Follows the reduction method of Kobayashi, Nakata, and Kojima (2007) but modernized for sparse matrices:
1. Detect and drop linearly dependent columns in `D` via sparse rank-revealing QR factorization.
2. Select a well-conditioned basis of rows `B` of `D` via column-pivoted sparse QR on `Dᵀ`.
3. Eliminate `z = -D_B⁻¹ (A_B(Z) + b_B)` from non-basis constraints `N` and the objective:
   - Constraints: `Ã(Z) + b̃ = 0` with `Ã = A_N - D_N D_B⁻¹ A_B` and `b̃ = b_N - D_N D_B⁻¹ b_B`.
   - Objective: `C̃ = C - A_Bᵀ D_B⁻ᵀ f` and `b̃₀ = b₀ - b_Bᵀ D_B⁻ᵀ f`.

Uses sparse LU factorizations to perform substitutions without explicitly forming dense inverse matrices.
Stores recovery data in `sdp.recovery_info` to reconstruct `z` via [`recover_affine_solution`](@ref).
"""
function presolve!(sdp::SemidefiniteProgram)
    N_triu = length(sdp.C)
    n = N_triu > 0 ? round(Int, (sqrt(8 * N_triu + 1) - 1) / 2) : 0
    p = length(sdp.f)
    m = size(sdp.A, 1)

    if sdp.config.verbose
        println("Starting presolve...")
        println("Problem dimensions: n = $n (N_triu = $N_triu), p = $p, m = $m")
    end

    if p == 0 || m == 0
        if sdp.config.verbose
            println("No affine variables or no constraints; skipping presolve.")
        end
        return
    end

    ######## STEP 1: Remove redundant columns in affine constraint matrix ########
    if sdp.config.verbose
        println("Performing first QR factorization find linearly independent columns in affine constraint matrix...")
    end
    @time_if sdp.config.verbose F1 = qr(sdp.D)
    R1 = F1.R
    diag_R1 = abs.(diag(R1))
    tol1 = eps(Float64) * max(m, p) * (isempty(diag_R1) ? 1.0 : maximum(diag_R1))
    nzdiag1 = findall(>(tol1), diag_R1)
    independent_cols = F1.pcol[nzdiag1]
    p_rank = length(independent_cols)

    if sdp.config.verbose
        println("Affine constraint matrix D has rank $p_rank out of $p columns.")
    end

    if p_rank == 0
        if sdp.config.verbose
            println("D has rank 0; all affine variables are redundant. Removing them.")
        end
        sdp.D = spzeros(Float64, m, 0)
        sdp.f = spzeros(Float64, 0)
        sdp.recovery_info = AffineRecoveryInfo(lu(zeros(Float64, 0, 0)), Float64[], spzeros(Float64, 0, 0), Int[], p)
        sdp.dual_recovery_info = DualRecoveryInfo(Int[], collect(1:m), lu(zeros(Float64, 0, 0)), Float64[], spzeros(Float64, m, 0), m)
        return
    end

    p_orig = p
    D_orig_indep = sdp.D[:, independent_cols]
    f_indep = Vector(sdp.f[independent_cols])

    sdp.D = copy(D_orig_indep)
    sdp.f = sparse(f_indep)
    p = p_rank

    ######## STEP 2: Extract basis from affine constraint matrix rows ########
    if sdp.config.verbose
        println("Performing second QR factorization to extract row space basis of affine constraint matrix...")
    end
    @time_if sdp.config.verbose F2 = qr(sparse(sdp.D'))
    B = F2.pcol[1:p]
    N = F2.pcol[p+1:end]

    if sdp.config.verbose
        println("Extracted basis rows B of size $(length(B)) and non-basis rows N of size $(length(N)).")
    end

    ######## STEP 3: Incorporate affine into conic data ########
    if sdp.config.verbose
        println("Incorporating affine variables into conic data...")
    end

    if sdp.config.verbose
        println("  Computing sparse LU factorization of basis matrix D[B, :]...")
    end
    @time_if sdp.config.verbose F_DB = lu(sdp.D[B, :])

    if sdp.config.verbose
        println("  Computing objective modification vector f_B...")
    end
    @time_if sdp.config.verbose f_B = F_DB' \ f_indep

    D_N = copy(sdp.D[N, :])
    b_B = Vector(sdp.b[B])
    A_B = sdp.A[B, :]

    sdp.recovery_info = AffineRecoveryInfo(F_DB, b_B, A_B, independent_cols, p_orig)
    sdp.dual_recovery_info = DualRecoveryInfo(B, N, F_DB, f_indep, D_N, m)

    sdp.b0 = sdp.b0 - dot(b_B, f_B)

    if sdp.config.verbose
        println("  Updating conic objective vector C...")
    end
    @time_if sdp.config.verbose sdp.C = sparse(sdp.C - A_B' * f_B)

    if sdp.config.verbose
        println("  Updating RHS vector b...")
    end
    @time_if sdp.config.verbose begin
        v_b = F_DB \ b_B
        sdp.b = sparse(sdp.b[N] - D_N * v_b)
    end

    nnz_DN = nnz(D_N)
    if sdp.config.verbose
        println("  Updating conic constraint matrix A (D_N nonzeros = $nnz_DN)...")
    end
    @time_if sdp.config.verbose begin
        if nnz_DN == 0
            if sdp.config.verbose
                println("    D_N has no nonzeros; non-basis constraints A[N, :] require no affine adjustments.")
            end
            sdp.A = sdp.A[N, :]
        else
            D_N_T = sparse(D_N')
            active_rows = findall(c -> D_N_T.colptr[c+1] > D_N_T.colptr[c], 1:length(N))
            if sdp.config.verbose
                println("    Found $(length(active_rows)) / $(length(N)) non-basis rows with nonzero affine interactions.")
            end

            W_I = Int[]
            W_J = Int[]
            W_V = Float64[]
            tol_zero = 1e-14

            for row_idx in active_rows
                d_col = Vector(D_N_T[:, row_idx])
                w_col = F_DB' \ d_col
                for j in 1:p
                    val = w_col[j]
                    if abs(val) > tol_zero
                        push!(W_I, row_idx)
                        push!(W_J, j)
                        push!(W_V, val)
                    end
                end
            end

            W = sparse(W_I, W_J, W_V, length(N), p)
            Delta_A = W * A_B
            sdp.A = sparse(sdp.A[N, :] - Delta_A)
        end
    end

    sdp.D = spzeros(Float64, length(N), 0)
    sdp.f = spzeros(Float64, 0)
end

"""
    recover_affine_solution(sdp::SemidefiniteProgram, Z_sol::AbstractVector{Float64})::Vector{Float64}

Recover the optimal affine variables `z ∈ ℝᵖ` from the flattened upper triangular solution
`Z_sol ∈ ℝᴺᵗʳⁱᵘ` of the presolved problem:

    z_B = -D_B⁻¹ (A_B Z_sol + b_B)

Dependent affine variables removed in step 1 are set to zero.
"""
function recover_affine_solution(sdp::SemidefiniteProgram, Z_sol::AbstractVector{Float64})::Vector{Float64}
    if isnothing(sdp.recovery_info)
        return Float64[]
    end
    info = sdp.recovery_info
    if isempty(info.independent_cols)
        return zeros(Float64, info.p_orig)
    end
    z_indep = -(info.F_DB \ (info.A_B * Z_sol + info.b_B))
    z = zeros(Float64, info.p_orig)
    z[info.independent_cols] = z_indep
    return z
end

"""
    recover_dual_solution(sdp::SemidefiniteProgram, y_presolved::AbstractVector{Float64})::Vector{Float64}

Recover the original equality constraint dual multipliers `y ∈ ℝᵐ` from the dual multipliers
`y_presolved ∈ ℝ|ᴺ|` of the presolved problem:

    y_N = y_presolved
    y_B = D_B⁻ᵀ (f_eff - D_Nᵀ y_presolved)

where `f_eff = f_indep` for `MOI.MIN_SENSE` and `f_eff = -f_indep` for `MOI.MAX_SENSE`.
If presolve was not performed or no affine variables were eliminated, returns `copy(y_presolved)`.
"""
function recover_dual_solution(sdp::SemidefiniteProgram, y_presolved::AbstractVector{Float64})::Vector{Float64}
    if isnothing(sdp.dual_recovery_info)
        return Vector{Float64}(y_presolved)
    end
    info = sdp.dual_recovery_info
    y = zeros(Float64, info.m)
    if !isempty(info.N)
        y[info.N] = y_presolved
    end
    if !isempty(info.B)
        f_eff = sdp.sense == MOI.MAX_SENSE ? -info.f_indep : info.f_indep
        rhs = if isempty(info.N) || isempty(y_presolved)
            f_eff
        else
            f_eff - info.D_N' * y_presolved
        end
        y[info.B] = info.F_DB' \ rhs
    end
    return y
end
