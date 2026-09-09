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

Presolve `sdp` in-place by eliminating free affine variables `z` (Stage 1) and optionally
linearly redundant conic equality constraints (Stage 2).

Follows the reduction method of Kobayashi, Nakata, and Kojima (2007) modernized for sparse matrices:
1. Detect and drop linearly dependent columns in `D` via sparse rank-revealing QR factorization.
2. Select a well-conditioned basis of rows `B` of `D` via column-pivoted sparse QR on `Dᵀ`.
3. Eliminate `z = -D_B⁻¹ (A_B(Z) + b_B)` from non-basis constraints `N` and the objective:
   - Constraints: `Ã(Z) + b̃ = 0` with `Ã = A_N - D_N D_B⁻¹ A_B` and `b̃ = b_N - D_N D_B⁻¹ b_B`.
   - Objective: `C̃ = C - A_Bᵀ D_B⁻ᵀ f` and `b̃₀ = b₀ - b_Bᵀ D_B⁻ᵀ f`.
4. If `eliminate_redundant_constraints` is enabled, factorize `Aᵀ` using column-pivoted sparse QR,
   check consistency against `b` (throwing `ErrorException("INFEASIBLE")` if inconsistent),
   and drop redundant constraints.

Controlled by `sdp.config`:
- `eliminate_free_variables`: If `true`, eliminates free affine variables `z` (default: `true`).
- `eliminate_redundant_constraints`: If `true`, detects and drops linearly redundant constraints in `A` (default: `false`).

Stores recovery data in `sdp.recovery_info` and `sdp.dual_recovery_info`.
"""
function presolve!(sdp::SemidefiniteProgram)
    N_triu = length(sdp.C)
    dim_desc = if !isempty(sdp.blocks)
        "$(length(sdp.blocks)) blocks, N_triu = $N_triu"
    else
        n = N_triu > 0 ? round(Int, (sqrt(8 * N_triu + 1) - 1) / 2) : 0
        "n = $n (N_triu = $N_triu)"
    end
    p = length(sdp.f)
    m = size(sdp.A, 1)

    if sdp.config.verbose
        println("Starting presolve...")
        println("Problem dimensions: $dim_desc, p = $p, m = $m")
        println("eliminate_free_variables: $(sdp.config.eliminate_free_variables), eliminate_redundant_constraints: $(sdp.config.eliminate_redundant_constraints)")
    end

    if m == 0
        if sdp.config.verbose
            println("No constraints; skipping presolve.")
        end
        return
    end

    # =========================================================================
    # STAGE 1: Elimination of free affine variables z
    # =========================================================================
    if sdp.config.eliminate_free_variables && p > 0
        ######## STEP 1: Remove redundant columns in affine constraint matrix ########
        if sdp.config.verbose
            println("Stage 1 - Step 1: Performing QR to find linearly independent columns in affine constraint matrix D...")
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
            sdp.dual_recovery_info = DualRecoveryInfo(Int[], collect(1:m), lu(zeros(Float64, 0, 0)), Float64[], spzeros(Float64, m, 0), m, Int[])
        else
            p_orig = p
            D_orig_indep = sdp.D[:, independent_cols]
            f_indep = Vector(sdp.f[independent_cols])

            sdp.D = copy(D_orig_indep)
            sdp.f = sparse(f_indep)
            p = p_rank

            ######## STEP 2: Extract basis from affine constraint matrix rows ########
            if sdp.config.verbose
                println("Stage 1 - Step 2: Performing QR to extract row space basis of affine constraint matrix D...")
            end
            @time_if sdp.config.verbose F2 = qr(sparse(sdp.D'))
            B = F2.pcol[1:p]
            N = F2.pcol[p+1:end]

            if sdp.config.verbose
                println("Extracted basis rows B of size $(length(B)) and non-basis rows N of size $(length(N)).")
            end

            ######## STEP 3: Incorporate affine into conic data ########
            if sdp.config.verbose
                println("Stage 1 - Step 3: Incorporating affine variables into conic data...")
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
            sdp.dual_recovery_info = DualRecoveryInfo(B, N, F_DB, f_indep, D_N, m, Int[])

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
    end

    # =========================================================================
    # STAGE 2: Elimination of linearly redundant conic equality constraints
    # =========================================================================
    if sdp.config.eliminate_redundant_constraints
        m_conic = size(sdp.A, 1)
        if m_conic > 0
            if sdp.config.verbose
                println("Stage 2: Detecting linearly redundant conic equality constraints in A (m = $m_conic)...")
            end
            @time_if sdp.config.verbose F_A = qr(sparse(sdp.A'))
            R_A = F_A.R
            diag_R_A = abs.(diag(R_A))
            tol_A = eps(Float64) * max(size(sdp.A)...) * (isempty(diag_R_A) ? 1.0 : maximum(diag_R_A))
            r_A = count(>(tol_A), diag_R_A)

            if sdp.config.verbose
                println("Stage 2: Conic constraint matrix A has rank $r_A out of $m_conic constraints.")
            end

            if r_A == 0
                if norm(sdp.b, Inf) > 1e-6
                    if sdp.config.verbose
                        println("Stage 2: A has rank 0 but b is nonzero! Infeasible.")
                    end
                    error("INFEASIBLE")
                end
                sdp.A = spzeros(Float64, 0, size(sdp.A, 2))
                sdp.b = spzeros(Float64, 0)
                if sdp.dual_recovery_info !== nothing
                    old_info = sdp.dual_recovery_info
                    sdp.dual_recovery_info = DualRecoveryInfo(
                        old_info.B,
                        old_info.N,
                        old_info.F_DB,
                        old_info.f_indep,
                        old_info.D_N,
                        old_info.m,
                        Int[]
                    )
                else
                    sdp.dual_recovery_info = DualRecoveryInfo(
                        Int[],
                        collect(1:m_conic),
                        lu(zeros(Float64, 0, 0)),
                        Float64[],
                        spzeros(Float64, m_conic, 0),
                        m_conic,
                        Int[]
                    )
                end
            elseif r_A < m_conic
                basis_rows = F_A.pcol[1:r_A]
                redundant_rows = F_A.pcol[r_A+1:end]

                # Pure sparse consistency check:
                # b_expected = W * b_basis = (R11 \ R12)' * b_basis = R12' * (R11' \ b_basis)
                # Solves a single sparse triangular system without forming dense matrices.
                R11 = UpperTriangular(R_A[1:r_A, 1:r_A])
                R12 = R_A[1:r_A, r_A+1:end]
                b_basis = Vector(sdp.b[basis_rows])
                v = R11' \ b_basis
                b_expected = R12' * v
                b_actual = Vector(sdp.b[redundant_rows])
                residual = norm(b_actual - b_expected, Inf)
                tol_infeas = max(1e-5, 1e-4 * norm(b_expected, Inf))

                if residual > tol_infeas
                    if sdp.config.verbose
                        println("Stage 2: Inconsistency detected in redundant constraints! Residual = $residual > tol $tol_infeas")
                    end
                    error("INFEASIBLE")
                end

                if sdp.config.verbose
                    println("Stage 2: Consistent! Eliminating $(m_conic - r_A) redundant constraints.")
                end

                sdp.A = sdp.A[basis_rows, :]
                sdp.b = sdp.b[basis_rows]

                # Update dual recovery info
                if sdp.dual_recovery_info !== nothing
                    old_info = sdp.dual_recovery_info
                    sdp.dual_recovery_info = DualRecoveryInfo(
                        old_info.B,
                        old_info.N,
                        old_info.F_DB,
                        old_info.f_indep,
                        old_info.D_N,
                        old_info.m,
                        basis_rows
                    )
                else
                    sdp.dual_recovery_info = DualRecoveryInfo(
                        Int[],
                        collect(1:m_conic),
                        lu(zeros(Float64, 0, 0)),
                        Float64[],
                        spzeros(Float64, m_conic, 0),
                        m_conic,
                        basis_rows
                    )
                end
            end
        end
    end
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

    # Un-pad Stage 2 (conic redundant constraints) if any were dropped
    y_N = if !isempty(info.conic_basis_rows)
        y_expanded = zeros(Float64, length(info.N))
        y_expanded[info.conic_basis_rows] = y_presolved
        y_expanded
    else
        Vector{Float64}(y_presolved)
    end

    if !isempty(info.N)
        y[info.N] = y_N
    end
    if !isempty(info.B)
        f_eff = sdp.sense == MOI.MAX_SENSE ? -info.f_indep : info.f_indep
        rhs = if isempty(info.N) || isempty(y_N)
            f_eff
        else
            f_eff - info.D_N' * y_N
        end
        y[info.B] = info.F_DB' \ rhs
    end
    return y
end
