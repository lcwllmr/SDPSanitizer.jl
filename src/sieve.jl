function get_unflatten_map(blocks::Vector{Int})
    n = sum(blocks)
    N_triu = sum(div(b * (b + 1), 2) for b in blocks; init=0)
    k_to_ij = Tuple{Int, Int}[]
    diag_to_k = zeros(Int, n)
    sizehint!(k_to_ij, N_triu)
    offset = 0
    k = 0
    for b in blocks
        for j in 1:b
            for i in 1:j
                k += 1
                push!(k_to_ij, (i + offset, j + offset))
                if i == j
                    diag_to_k[i + offset] = k
                end
            end
        end
        offset += b
    end
    return n, N_triu, k_to_ij, diag_to_k
end

"""
    sieve!(sdp::SemidefiniteProgram; max_iter::Int=10, tol::Float64=1e-12)

Apply the Sieve-SDP facial reduction algorithm to the given `sdp`.
"""
function sieve!(sdp::SemidefiniteProgram; max_iter::Int=10, tol::Float64=1e-12)
    p = size(sdp.D, 2)
    if isempty(sdp.blocks)
        # Sieve cannot proceed without block structure.
        if sdp.config.verbose
            println("[Sieve] No block structure provided, skipping Sieve.")
        end
        return collect(1:size(sdp.A, 1))
    end

    n, N_triu, k_to_ij, diag_to_k = get_unflatten_map(sdp.blocks)
    m = size(sdp.A, 1)

    if m == 0
        return Int[]
    end

    active_rows = trues(n)
    undeleted_constraints = trues(m)
    
    A_T = sparse(sdp.A')
    rows_A_T = rowvals(A_T)
    vals_A_T = nonzeros(A_T)

    undone = true
    iter = 0

    if sdp.config.verbose
        println("[Sieve] Starting Sieve-SDP on $n variables, $m constraints.")
    end

    D_T = sparse(sdp.D')
    
    while undone && iter < max_iter
        undone = false
        iter += 1

        for c in 1:m
            if !undeleted_constraints[c]
                continue
            end
            
            has_affine = false
            if p > 0
                for ptr in nzrange(D_T, c)
                    if abs(nonzeros(D_T)[ptr]) > tol
                        has_affine = true
                        break
                    end
                end
            end
            if has_affine
                continue
            end

            I_aux = falses(n)
            for ptr in nzrange(A_T, c)
                k = rows_A_T[ptr]
                val = vals_A_T[ptr]
                if abs(val) > tol
                    i, j = k_to_ij[k]
                    if active_rows[i] && active_rows[j]
                        I_aux[i] = true
                        I_aux[j] = true
                    end
                end
            end

            idx = findall(I_aux)
            nnz_sub = length(idx)
            
            b_c = sdp.b[c]

            if nnz_sub == 0
                if abs(b_c) > tol
                    error("INFEASIBLE") 
                else
                    undeleted_constraints[c] = false
                    undone = true
                end
                continue
            end

            At = zeros(Float64, nnz_sub, nnz_sub)
            for ptr in nzrange(A_T, c)
                k = rows_A_T[ptr]
                val = vals_A_T[ptr]
                if abs(val) > tol
                    i, j = k_to_ij[k]
                    if active_rows[i] && active_rows[j]
                        i_sub = findfirst(==(i), idx)
                        j_sub = findfirst(==(j), idx)
                        actual_val = (i == j) ? val : val / sqrt(2.0)
                        At[i_sub, j_sub] = actual_val
                        At[j_sub, i_sub] = actual_val
                    end
                end
            end

            is_pd = isposdef(Symmetric(At))
            is_nd = isposdef(Symmetric(-At))
            
            if is_pd
                if b_c > tol
                    error("INFEASIBLE")
                elseif abs(b_c) <= tol
                    active_rows[idx] .= false
                    undeleted_constraints[c] = false
                    undone = true
                end
            elseif is_nd
                if b_c < -tol
                    error("INFEASIBLE")
                elseif abs(b_c) <= tol
                    active_rows[idx] .= false
                    undeleted_constraints[c] = false
                    undone = true
                end
            end
        end
    end

    if sdp.config.verbose
        println("[Sieve] Finished after $iter iterations. Active variables: $(sum(active_rows)) / $n.")
    end

    active_con_idx = findall(undeleted_constraints)
    sdp.A = sdp.A[active_con_idx, :]
    sdp.D = sdp.D[active_con_idx, :]
    sdp.b = sdp.b[active_con_idx]

    deleted_rows = findall(.~active_rows)
    if !isempty(deleted_rows)
        diag_cols = [diag_to_k[i] for i in deleted_rows]
        
        m_new = size(sdp.A, 1)
        m_add = length(diag_cols)
        
        I_add = collect(1:m_add)
        J_add = diag_cols
        V_add = ones(Float64, m_add)
        A_add = sparse(I_add, J_add, V_add, m_add, N_triu)
        
        sdp.A = vcat(sdp.A, A_add)
        sdp.D = vcat(sdp.D, spzeros(Float64, m_add, p))
        sdp.b = vcat(sdp.b, spzeros(Float64, m_add))
    end
    
    return active_con_idx
end
