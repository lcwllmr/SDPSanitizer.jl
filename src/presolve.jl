macro time_if(cond, expr)
    quote
        if $(esc(cond))
            @time $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

function get_effective_threads(config_threads::Int)::Tuple{Int, Int}
    if config_threads <= 0
        j_threads = max(1, min(Threads.nthreads(), 8))
        spqr_threads = max(1, min(Sys.CPU_THREADS, 8))
        return j_threads, spqr_threads
    else
        j_threads = max(1, min(config_threads, Threads.nthreads()))
        spqr_threads = max(1, min(config_threads, Sys.CPU_THREADS))
        return j_threads, spqr_threads
    end
end

function configure_spqr_threads!(spqr_threads::Int)
    ss = get(Base.loaded_modules, Base.PkgId(Base.UUID("4607b0f0-06f3-5cda-b6b1-a6196a1729e9"), "SuiteSparse"), nothing)
    if ss !== nothing && isdefined(ss, :SPQR) && isdefined(ss.SPQR, :set_spqr_nthreads)
        try
            ss.SPQR.set_spqr_nthreads(spqr_threads)
        catch
        end
    end
end

function _solve_w_chunk!(
    I_t::Vector{Int},
    J_t::Vector{Int},
    V_t::Vector{Float64},
    k_range::UnitRange{Int},
    active_rows::Vector{Int},
    D_N_T::SparseMatrixCSC{Float64, Int},
    F_DB,
    p::Int,
    tol_zero::Float64
)
    F_task = copy(F_DB)
    w_buf = zeros(Float64, p)
    d_buf = zeros(Float64, p)

    for k in k_range
        row_idx = active_rows[k]
        for ptr in nzrange(D_N_T, row_idx)
            d_buf[D_N_T.rowval[ptr]] = D_N_T.nzval[ptr]
        end
        ldiv!(w_buf, F_task', d_buf)
        for ptr in nzrange(D_N_T, row_idx)
            d_buf[D_N_T.rowval[ptr]] = 0.0
        end
        for j in 1:p
            val = w_buf[j]
            if abs(val) > tol_zero
                push!(I_t, row_idx)
                push!(J_t, j)
                push!(V_t, val)
            end
        end
    end
end

function _compute_a_chunk!(
    A_chunks::Vector{SparseMatrixCSC{Float64, Int}},
    t_idx::Int,
    r_range::UnitRange{Int},
    W::SparseMatrixCSC{Float64, Int},
    A_N_full::SparseMatrixCSC{Float64, Int},
    A_B::SparseMatrixCSC{Float64, Int}
)
    W_sub = W[r_range, :]
    A_orig_sub = A_N_full[r_range, :]
    if nnz(W_sub) == 0
        A_chunks[t_idx] = A_orig_sub
    else
        Delta_sub = W_sub * A_B
        diff = A_orig_sub - Delta_sub
        max_diff = isempty(diff.nzval) ? 0.0 : maximum(abs.(diff.nzval))
        if max_diff > 1e15
            max_w_sub = isempty(W_sub.nzval) ? 0.0 : maximum(abs.(W_sub.nzval))
            max_ab = isempty(A_B.nzval) ? 0.0 : maximum(abs.(A_B.nzval))
            println("    [Chunk $t_idx Alert] max |diff| = $max_diff, max |W_sub| = $max_w_sub, max |A_B| = $max_ab")
        end
        tol_noise = 1e-12
        for ptr in 1:nnz(diff)
            if abs(diff.nzval[ptr]) < tol_noise
                diff.nzval[ptr] = 0.0
            end
        end
        A_chunks[t_idx] = dropzeros!(diff)
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
- `num_threads`: Number of worker threads. `0` auto-selects up to min(available, 8) (default: `0`).

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

    j_threads, spqr_threads = get_effective_threads(sdp.config.num_threads)
    configure_spqr_threads!(spqr_threads)

    if sdp.config.verbose
        println("Starting presolve...")
        println("Problem dimensions: $dim_desc, p = $p, m = $m")
        println("Worker threads: Julia = $j_threads, SPQR = $spqr_threads")
        println("eliminate_free_variables: $(sdp.config.eliminate_free_variables), eliminate_redundant_constraints: $(sdp.config.eliminate_redundant_constraints)")
        max_d = isempty(sdp.D.nzval) ? 0.0 : maximum(abs.(sdp.D.nzval))
        println("sdp.D max entry before presolve: $max_d")
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
        D_col_scaled = copy(sdp.D)
        for j in 1:p
            s = 0.0
            for ptr in nzrange(D_col_scaled, j)
                s = hypot(s, D_col_scaled.nzval[ptr])
            end
            if s > 0.0
                for ptr in nzrange(D_col_scaled, j)
                    D_col_scaled.nzval[ptr] /= s
                end
            end
        end
        @time_if sdp.config.verbose F1 = qr(D_col_scaled)
        R1 = F1.R::SparseMatrixCSC{Float64, Int}
        diag_R1 = abs.(diag(R1))
        max_diag1 = isempty(diag_R1) ? 1.0 : maximum(diag_R1)
        tol1 = max(eps(Float64) * max(m, p) * max_diag1, sdp.config.cond_tol * max_diag1)
        nzdiag1 = findall(>(tol1), diag_R1)
        independent_cols = (F1.pcol::Vector{Int})[nzdiag1]
        p_rank = length(independent_cols)

        if sdp.config.verbose
            println("Affine constraint matrix D has rank $p_rank out of $p columns.")
            if !isempty(diag_R1)
                println("  R1 pivots: min = $(minimum(diag_R1)), max = $max_diag1, tol1 = $tol1")
            end
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
            if p_rank == p && independent_cols == collect(1:p)
                f_indep = Vector(sdp.f)
            else
                D_orig_indep = sdp.D[:, independent_cols]
                f_indep = Vector(sdp.f[independent_cols])

                sdp.D = copy(D_orig_indep)
                sdp.f = sparse(f_indep)
                D_col_scaled = D_col_scaled[:, independent_cols]
                p = p_rank
            end

            ######## STEP 2: Extract basis from affine constraint matrix rows ########
            if sdp.config.verbose
                println("Stage 1 - Step 2: Performing QR to extract row space basis of affine constraint matrix D...")
            end
            # Column equilibration: normalize each column by its maximum magnitude
            # so that relative magnitudes are balanced without deleting genuine nonzeros.
            D_piv = copy(sdp.D)
            for j in 1:p
                cmax = 0.0
                for ptr in nzrange(D_piv, j)
                    v = abs(D_piv.nzval[ptr])
                    if v > cmax
                        cmax = v
                    end
                end
                if cmax > 0.0
                    for ptr in nzrange(D_piv, j)
                        D_piv.nzval[ptr] /= cmax
                    end
                end
            end
            @time_if sdp.config.verbose F2 = qr(sparse(D_piv'))
            R2 = F2.R::SparseMatrixCSC{Float64, Int}
            diag_R2 = abs.(diag(R2))
            max_diag2 = isempty(diag_R2) ? 1.0 : maximum(diag_R2)
            tol2 = max(eps(Float64) * max(m, p) * max_diag2, sdp.config.cond_tol * max_diag2)
            nzdiag2 = findall(>(tol2), diag_R2)
            pcol2 = F2.pcol::Vector{Int}
            n_pivots = min(p, length(nzdiag2))
            B = pcol2[nzdiag2[1:n_pivots]]
            D_piv = spzeros(Float64, 0, 0)
            D_col_scaled = spzeros(Float64, 0, 0)

            in_B = falses(m)
            in_B[B] .= true
            N = findall(!, in_B)

            if sdp.config.verbose
                println("Extracted basis rows B of size $(length(B)) and non-basis rows N of size $(length(N)).")
                if !isempty(diag_R2)
                    println("  R2 pivots: min = $(minimum(diag_R2)), max = $max_diag2, tol2 = $tol2")
                end
            end

            ######## STEP 3: Incorporate affine into conic data ########
            if sdp.config.verbose
                println("Stage 1 - Step 3: Incorporating affine variables into conic data...")
            end

            if sdp.config.verbose
                println("  Computing sparse LU factorization of basis matrix D[B, :] with two-sided equilibration...")
            end
            D_B_raw = sdp.D[B, :]
            c_scale = zeros(Float64, p)
            for j in 1:p
                cmax = 0.0
                for ptr in nzrange(D_B_raw, j)
                    v = abs(D_B_raw.nzval[ptr])
                    if v > cmax
                        cmax = v
                    end
                end
                c_scale[j] = cmax > 0.0 ? 1.0 / cmax : 1.0
            end

            r_scale = zeros(Float64, length(B))
            for j in 1:p
                cj = c_scale[j]
                for ptr in nzrange(D_B_raw, j)
                    row = D_B_raw.rowval[ptr]
                    v = abs(D_B_raw.nzval[ptr]) * cj
                    if v > r_scale[row]
                        r_scale[row] = v
                    end
                end
            end
            for i in 1:length(r_scale)
                r_scale[i] = r_scale[i] > 0.0 ? 1.0 / r_scale[i] : 1.0
            end

            D_B_equil = copy(D_B_raw)
            for j in 1:p
                cj = c_scale[j]
                for ptr in nzrange(D_B_equil, j)
                    row = D_B_equil.rowval[ptr]
                    D_B_equil.nzval[ptr] *= (r_scale[row] * cj)
                end
            end

            @time_if sdp.config.verbose raw_lu = lu(D_B_equil)
            if sdp.config.verbose
                U_diag = abs.(diag(raw_lu.U))
                min_u = isempty(U_diag) ? 1.0 : minimum(U_diag)
                max_u = isempty(U_diag) ? 1.0 : maximum(U_diag)
                println("  D_B LU U-diagonal pivots: min = $min_u, max = $max_u, cond_est = $(max_u / min_u)")
                println("    Pivots < 1e-12: $(count(<(1e-12), U_diag)), < 1e-8: $(count(<(1e-8), U_diag)), < 1e-5: $(count(<(1e-5), U_diag))")
                println("    r_scale: min = $(minimum(r_scale)), max = $(maximum(r_scale))")
                println("    c_scale: min = $(minimum(c_scale)), max = $(maximum(c_scale))")
            end
            F_DB = EquilibratedLU(raw_lu, r_scale, c_scale)

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
                println("  Updating conic constraint matrix A (D_N nonzeros = $nnz_DN, threads = $j_threads)...")
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
                    n_active = length(active_rows)
                    if sdp.config.verbose
                        println("    Found $n_active / $(length(N)) non-basis rows with nonzero affine interactions.")
                    end

                    tol_zero = 1e-14
                    n_tasks_w = min(j_threads, n_active)

                    if n_tasks_w <= 1
                        W_I = Int[]
                        W_J = Int[]
                        W_V = Float64[]
                        _solve_w_chunk!(W_I, W_J, W_V, 1:n_active, active_rows, D_N_T, F_DB, p, tol_zero)
                        W = sparse(W_I, W_J, W_V, length(N), p)
                    else
                        chunk_size_w = cld(n_active, n_tasks_w)
                        partitions_w = collect(Iterators.partition(1:n_active, chunk_size_w))
                        thread_triplets = [ (Int[], Int[], Float64[]) for _ in 1:length(partitions_w) ]

                        @sync for t_idx in 1:length(partitions_w)
                            let t_idx = t_idx, k_range = UnitRange{Int}(partitions_w[t_idx])
                                Threads.@spawn begin
                                    I_t, J_t, V_t = thread_triplets[t_idx]
                                    _solve_w_chunk!(I_t, J_t, V_t, k_range, active_rows, D_N_T, F_DB, p, tol_zero)
                                end
                            end
                        end

                        total_nnz_w = sum(length(t[1]) for t in thread_triplets)
                        W_I = Vector{Int}(undef, total_nnz_w)
                        W_J = Vector{Int}(undef, total_nnz_w)
                        W_V = Vector{Float64}(undef, total_nnz_w)
                        offset = 0
                        for t in thread_triplets
                            len = length(t[1])
                            copyto!(W_I, offset + 1, t[1], 1, len)
                            copyto!(W_J, offset + 1, t[2], 1, len)
                            copyto!(W_V, offset + 1, t[3], 1, len)
                            offset += len
                        end
                        W = sparse(W_I, W_J, W_V, length(N), p)
                    end

                    n_N = length(N)
                    A_N_full = sdp.A[N, :]
                    n_tasks_a = min(j_threads, n_N)

                    if n_tasks_a <= 1
                        Delta_A = W * A_B
                        if sdp.config.verbose
                            max_w = isempty(W.nzval) ? 0.0 : maximum(abs.(W.nzval))
                            max_ab = isempty(A_B.nzval) ? 0.0 : maximum(abs.(A_B.nzval))
                            max_delta = isempty(Delta_A.nzval) ? 0.0 : maximum(abs.(Delta_A.nzval))
                            println("    [Stage 1 Diagnostic] max |W| = $max_w, max |A_B| = $max_ab, max |Delta_A| = $max_delta")
                        end
                        diff = A_N_full - Delta_A
                        tol_noise = 1e-12
                        for ptr in 1:nnz(diff)
                            if abs(diff.nzval[ptr]) < tol_noise
                                diff.nzval[ptr] = 0.0
                            end
                        end
                        sdp.A = dropzeros!(diff)
                    else
                        chunk_size_a = cld(n_N, n_tasks_a)
                        partitions_a = collect(Iterators.partition(1:n_N, chunk_size_a))
                        A_chunks = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(partitions_a))

                        @sync for t_idx in 1:length(partitions_a)
                            let t_idx = t_idx, r_range = UnitRange{Int}(partitions_a[t_idx])
                                Threads.@spawn begin
                                    _compute_a_chunk!(A_chunks, t_idx, r_range, W, A_N_full, A_B)
                                end
                            end
                        end
                        sdp.A = vcat(A_chunks...)
                    end
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
