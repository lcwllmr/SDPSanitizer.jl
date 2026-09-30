# MathOptInterface wrapper for SDPSanitizer

"""
    MOIWrapper(inner; presolve::Bool = true, verbose::Bool = false)

An MOI optimizer wrapper that intercepts semidefinite programs, optionally applies
the SDPSanitizer presolve routine to eliminate free affine variables, passes the
reduced problem to `inner`, and recovers the primal and dual solutions
for the original formulation.

`inner` can be an instantiated `MOI.ModelLike` or an optimizer constructor function/type.
"""
mutable struct MOIWrapper{O <: MOI.ModelLike} <: MOI.AbstractOptimizer
    inner::O
    presolve::Bool
    eliminate_free_variables::Bool
    eliminate_redundant_constraints::Bool
    detect_infeasibility::Bool
    facial_reduction::Bool
    facial_reduction_applied::Bool
    num_threads::Int
    cond_tol::Float64
    verbose::Bool
    silent::Bool

    infeasible::Bool
    infeasible_msg::String
    sieve_active_con_idx::Union{Nothing, Vector{Int}}

    model::MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}
    sdp::Union{Nothing, SemidefiniteProgram}

    # Variable classification
    psd_vars::Vector{MOI.VariableIndex}
    affine_vars::Vector{MOI.VariableIndex}
    psd_var_to_col::Dict{MOI.VariableIndex, Int}
    affine_var_to_col::Dict{MOI.VariableIndex, Int}

    # Constraint tracking
    psd_con_indices::Vector{Tuple{MOI.ConstraintIndex, Int}} # (ci, dimension)
    nonneg_con_indices::Vector{MOI.ConstraintIndex}
    gt_con_indices::Vector{MOI.ConstraintIndex}
    affine_con_indices::Vector{MOI.ConstraintIndex}
    affine_con_dims::Vector{Int}
    con_row_offsets::Dict{MOI.ConstraintIndex, Int}
    m_total::Int

    # Inner model map
    inner_var_map::Dict{MOI.VariableIndex, MOI.VariableIndex}
    inner_psd_con_map::Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}
    inner_affine_con_map::Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}
    inner_presolved_con::Union{Nothing, MOI.ConstraintIndex}

    # Cached solutions
    recovered_affine_primal::Vector{Float64}
    recovered_dual::Vector{Float64}

    # Timings
    solve_time::Float64
    inner_solve_time::Float64

    function MOIWrapper(inner::O;
        presolve::Bool = true,
        eliminate_free_variables::Bool = true,
        eliminate_redundant_constraints::Bool = false,
        detect_infeasibility::Bool = false,
        facial_reduction::Bool = false,
        sieve::Union{Nothing, Bool} = nothing,
        num_threads::Int = 0,
        cond_tol::Float64 = 1e-9,
        verbose::Bool = false
    ) where {O <: MOI.ModelLike}
        bridged_inner = if inner isa MOI.Bridges.AbstractBridgeOptimizer
            inner
        else
            cache = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
            caching = MOI.Utilities.CachingOptimizer(cache, inner)
            MOI.Bridges.full_bridge_optimizer(caching, Float64)
        end
        silent = false
        try
            silent = MOI.get(bridged_inner, MOI.Silent())
        catch
        end

        eff_detect_infeas = sieve !== nothing ? sieve : detect_infeasibility
        eff_facial_red = sieve !== nothing ? sieve : facial_reduction

        return new{typeof(bridged_inner)}(
            bridged_inner,
            presolve,
            eliminate_free_variables,
            eliminate_redundant_constraints,
            eff_detect_infeas,
            eff_facial_red,
            false, # facial_reduction_applied
            num_threads,
            cond_tol,
            verbose,
            silent,
            false, # infeasible
            "", # infeasible_msg
            nothing, # sieve_active_con_idx
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            nothing,
            MOI.VariableIndex[],
            MOI.VariableIndex[],
            Dict{MOI.VariableIndex, Int}(),
            Dict{MOI.VariableIndex, Int}(),
            Tuple{MOI.ConstraintIndex, Int}[],
            MOI.ConstraintIndex[],
            MOI.ConstraintIndex[],
            MOI.ConstraintIndex[],
            Int[],
            Dict{MOI.ConstraintIndex, Int}(),
            0,
            Dict{MOI.VariableIndex, MOI.VariableIndex}(),
            Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}(),
            Dict{MOI.ConstraintIndex, MOI.ConstraintIndex}(),
            nothing,
            Float64[],
            Float64[],
            0.0, # solve_time
            0.0  # inner_solve_time
        )
    end
end

const LAST_INNER_SOLVE_TIME = Ref{Float64}(0.0)
const LAST_TOTAL_SOLVE_TIME = Ref{Float64}(0.0)

get_last_inner_solve_time() = LAST_INNER_SOLVE_TIME[]
get_last_total_solve_time() = LAST_TOTAL_SOLVE_TIME[]

function reset_solve_times!()
    LAST_INNER_SOLVE_TIME[] = 0.0
    LAST_TOTAL_SOLVE_TIME[] = 0.0
    return nothing
end

MOIWrapper(optimizer_constructor::Function; kwargs...) = MOIWrapper(optimizer_constructor(); kwargs...)
MOIWrapper(optimizer_type::Type{<:MOI.ModelLike}; kwargs...) = MOIWrapper(optimizer_type(); kwargs...)

MOI.is_empty(opt::MOIWrapper) = MOI.is_empty(opt.model)

function MOI.empty!(opt::MOIWrapper)
    MOI.empty!(opt.model)
    MOI.empty!(opt.inner)
    opt.sdp = nothing
    empty!(opt.psd_vars)
    empty!(opt.affine_vars)
    empty!(opt.psd_var_to_col)
    empty!(opt.affine_var_to_col)
    empty!(opt.psd_con_indices)
    empty!(opt.nonneg_con_indices)
    empty!(opt.gt_con_indices)
    empty!(opt.affine_con_indices)
    empty!(opt.affine_con_dims)
    empty!(opt.con_row_offsets)
    opt.m_total = 0
    empty!(opt.inner_var_map)
    empty!(opt.inner_psd_con_map)
    empty!(opt.inner_affine_con_map)
    opt.inner_presolved_con = nothing
    opt.infeasible = false
    opt.infeasible_msg = ""
    opt.sieve_active_con_idx = nothing
    opt.facial_reduction_applied = false
    empty!(opt.recovered_affine_primal)
    empty!(opt.recovered_dual)
end

# Capabilities
MOI.supports_incremental_interface(::MOIWrapper) = true
MOI.supports_constraint(::MOIWrapper, ::Type{F}, ::Type{S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet} = true
MOI.supports_add_constrained_variables(::MOIWrapper, ::Type{MOI.Reals}) = true
MOI.supports_add_constrained_variables(::MOIWrapper, ::Type{S}) where {S <: MOI.AbstractVectorSet} = true
MOI.supports_add_constrained_variable(::MOIWrapper, ::Type{S}) where {S <: MOI.AbstractScalarSet} = true
MOI.supports(::MOIWrapper, attr::MOI.AbstractOptimizerAttribute) = true
MOI.supports(::MOIWrapper, attr::MOI.AbstractModelAttribute) = true
MOI.supports(::MOIWrapper, attr::MOI.AbstractVariableAttribute, ::Type{MOI.VariableIndex}) = true
MOI.supports(::MOIWrapper, attr::MOI.AbstractConstraintAttribute, ::Type{<:MOI.ConstraintIndex}) = true

# Model construction pass-through to internal model cache
MOI.add_variable(opt::MOIWrapper) = MOI.add_variable(opt.model)
MOI.add_variables(opt::MOIWrapper, n::Int) = MOI.add_variables(opt.model, n)
MOI.add_constraint(opt::MOIWrapper, f::MOI.AbstractFunction, s::MOI.AbstractSet) = MOI.add_constraint(opt.model, f, s)
MOI.add_constrained_variables(opt::MOIWrapper, s::MOI.AbstractVectorSet) = MOI.add_constrained_variables(opt.model, s)
MOI.add_constrained_variable(opt::MOIWrapper, s::MOI.AbstractScalarSet) = MOI.add_constrained_variable(opt.model, s)

# Attribute setters
MOI.set(opt::MOIWrapper, attr::MOI.Silent, val::Bool) = (opt.silent = val; MOI.set(opt.inner, attr, val))
MOI.set(opt::MOIWrapper, attr::MOI.ObjectiveSense, val) = MOI.set(opt.model, attr, val)
MOI.set(opt::MOIWrapper, attr::MOI.ObjectiveFunction, val) = MOI.set(opt.model, attr, val)
MOI.set(opt::MOIWrapper, attr::MOI.AbstractModelAttribute, val) = MOI.set(opt.model, attr, val)
MOI.set(opt::MOIWrapper, attr::MOI.AbstractVariableAttribute, v::MOI.VariableIndex, val) = MOI.set(opt.model, attr, v, val)
MOI.set(opt::MOIWrapper, attr::MOI.AbstractConstraintAttribute, c::MOI.ConstraintIndex, val) = MOI.set(opt.model, attr, c, val)

function MOI.set(opt::MOIWrapper, attr::MOI.RawOptimizerAttribute, val)
    if attr.name == "presolve"
        opt.presolve = Bool(val)
    elseif attr.name == "eliminate_free_variables"
        opt.eliminate_free_variables = Bool(val)
    elseif attr.name == "eliminate_redundant_constraints"
        opt.eliminate_redundant_constraints = Bool(val)
    elseif attr.name == "detect_infeasibility"
        opt.detect_infeasibility = Bool(val)
    elseif attr.name == "facial_reduction"
        opt.facial_reduction = Bool(val)
    elseif attr.name == "sieve"
        opt.detect_infeasibility = Bool(val)
        opt.facial_reduction = Bool(val)
    elseif attr.name == "num_threads" || attr.name == "threads"
        opt.num_threads = Int(val)
    elseif attr.name == "cond_tol"
        opt.cond_tol = Float64(val)
    elseif attr.name == "verbose"
        opt.verbose = Bool(val)
    else
        MOI.set(opt.inner, attr, val)
    end
end

MOI.set(opt::MOIWrapper, attr::MOI.AbstractOptimizerAttribute, val) = MOI.set(opt.inner, attr, val)

# Copying input model
function MOI.copy_to(dest::MOIWrapper, src::MOI.ModelLike)
    MOI.empty!(dest)
    return MOI.copy_to(dest.model, src)
end

# Optimization and presolve pipeline
function MOI.optimize!(opt::MOIWrapper)
    reset_solve_times!()
    t_start = time()
    # 1. Classify variables: Conic PSD vs free affine variables
    all_vars = MOI.get(opt.model, MOI.ListOfVariableIndices())
    psd_var_set = Set{MOI.VariableIndex}()

    empty!(opt.psd_vars)
    empty!(opt.affine_vars)
    empty!(opt.psd_var_to_col)
    empty!(opt.affine_var_to_col)
    empty!(opt.psd_con_indices)
    empty!(opt.nonneg_con_indices)
    empty!(opt.gt_con_indices)
    empty!(opt.affine_con_indices)
    empty!(opt.affine_con_dims)
    empty!(opt.con_row_offsets)
    empty!(opt.inner_var_map)
    empty!(opt.inner_psd_con_map)
    opt.inner_presolved_con = nothing
    empty!(opt.recovered_affine_primal)
    empty!(opt.recovered_dual)

    conic_var_set = Set{MOI.VariableIndex}()
    blocks = Int[]

    for (F, S) in MOI.get(opt.model, MOI.ListOfConstraintTypesPresent())
        if S <: Union{MOI.PositiveSemidefiniteConeTriangle, MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}}
            for c in MOI.get(opt.model, MOI.ListOfConstraintIndices{F, S}())
                s = MOI.get(opt.model, MOI.ConstraintSet(), c)
                dim = MOI.dimension(s)
                push!(opt.psd_con_indices, (c, dim))
                f = MOI.get(opt.model, MOI.ConstraintFunction(), c)
                vars = f isa MOI.VectorOfVariables ? f.variables : [t.scalar_term.variable for t in f.terms]
                added = false
                for v in vars
                    if !(v in conic_var_set)
                        push!(opt.psd_vars, v)
                        push!(conic_var_set, v)
                        added = true
                    end
                end
                if added
                    push!(blocks, round(Int, (sqrt(8 * dim + 1) - 1) / 2))
                end
            end
        elseif S <: MOI.Nonnegatives
            for c in MOI.get(opt.model, MOI.ListOfConstraintIndices{F, S}())
                push!(opt.nonneg_con_indices, c)
                f = MOI.get(opt.model, MOI.ConstraintFunction(), c)
                vars = f isa MOI.VectorOfVariables ? f.variables : [t.scalar_term.variable for t in f.terms]
                count = 0
                for v in vars
                    if !(v in conic_var_set)
                        push!(opt.psd_vars, v)
                        push!(conic_var_set, v)
                        count += 1
                    end
                end
                append!(blocks, ones(Int, count))
            end
        elseif S <: MOI.GreaterThan
            for c in MOI.get(opt.model, MOI.ListOfConstraintIndices{F, S}())
                s = MOI.get(opt.model, MOI.ConstraintSet(), c)
                if s.lower >= 0.0
                    push!(opt.gt_con_indices, c)
                    f = MOI.get(opt.model, MOI.ConstraintFunction(), c)
                    v = f isa MOI.VariableIndex ? f : f.variable
                    if !(v in conic_var_set)
                        push!(opt.psd_vars, v)
                        push!(conic_var_set, v)
                        push!(blocks, 1)
                    end
                end
            end
        end
    end

    for v in all_vars
        if !(v in conic_var_set)
            push!(opt.affine_vars, v)
        end
    end

    for (i, v) in enumerate(opt.psd_vars)
        opt.psd_var_to_col[v] = i
    end
    for (i, v) in enumerate(opt.affine_vars)
        opt.affine_var_to_col[v] = i
    end

    n = length(opt.psd_vars)
    p = length(opt.affine_vars)

    if opt.verbose
        println("[SDPSanitizer.MOIWrapper] Found $n conic variables and $p affine variables.")
        println("[SDPSanitizer.MOIWrapper] Constraint types present: ", MOI.get(opt.model, MOI.ListOfConstraintTypesPresent()))
    end

    # 2. Extract equality constraints
    row_offset = 0
    for (F, S) in MOI.get(opt.model, MOI.ListOfConstraintTypesPresent())
        if S <: MOI.Zeros
            for c in MOI.get(opt.model, MOI.ListOfConstraintIndices{F, S}())
                dim = MOI.dimension(MOI.get(opt.model, MOI.ConstraintSet(), c))
                push!(opt.affine_con_indices, c)
                push!(opt.affine_con_dims, dim)
                opt.con_row_offsets[c] = row_offset
                row_offset += dim
            end
        elseif S <: MOI.EqualTo
            for c in MOI.get(opt.model, MOI.ListOfConstraintIndices{F, S}())
                push!(opt.affine_con_indices, c)
                push!(opt.affine_con_dims, 1)
                opt.con_row_offsets[c] = row_offset
                row_offset += 1
            end
        end
    end
    opt.m_total = row_offset
    m = opt.m_total

    # Check if both presolve and sieve should be skipped
    has_presolve_work = opt.presolve && ((opt.eliminate_free_variables && p > 0) || opt.eliminate_redundant_constraints)
    has_sieve_work = opt.detect_infeasibility || opt.facial_reduction
    if !has_presolve_work && !has_sieve_work
        if opt.verbose
            println("[SDPSanitizer.MOIWrapper] Presolve not active or no presolve work ($p free variables), and sieve/infeasibility detection not active; passing directly to inner solver.")
        end
        index_map = MOI.copy_to(opt.inner, opt.model)
        for v in all_vars
            opt.inner_var_map[v] = index_map[v]
        end
        for c in opt.affine_con_indices
            if haskey(index_map, c)
                opt.inner_affine_con_map[c] = index_map[c]
            end
        end
        MOI.optimize!(opt.inner)
        return
    end

    # 3. Assemble A, D, b for equality constraints A(Z) + D z + b = 0
    A_I = Int[]; A_J = Int[]; A_V = Float64[]
    D_I = Int[]; D_J = Int[]; D_V = Float64[]
    b_vec = zeros(Float64, m)

    for c in opt.affine_con_indices
        offset = opt.con_row_offsets[c]
        f_con = MOI.get(opt.model, MOI.ConstraintFunction(), c)
        s_con = MOI.get(opt.model, MOI.ConstraintSet(), c)
        if f_con isa MOI.VectorAffineFunction
            for t in f_con.terms
                row = offset + t.output_index
                v = t.scalar_term.variable
                coeff = t.scalar_term.coefficient
                if haskey(opt.psd_var_to_col, v)
                    push!(A_I, row); push!(A_J, opt.psd_var_to_col[v]); push!(A_V, coeff)
                elseif haskey(opt.affine_var_to_col, v)
                    push!(D_I, row); push!(D_J, opt.affine_var_to_col[v]); push!(D_V, coeff)
                end
            end
            for (i, val) in enumerate(f_con.constants)
                b_vec[offset + i] = val
            end
        elseif f_con isa MOI.ScalarAffineFunction
            for t in f_con.terms
                v = t.variable
                coeff = t.coefficient
                if haskey(opt.psd_var_to_col, v)
                    push!(A_I, offset + 1); push!(A_J, opt.psd_var_to_col[v]); push!(A_V, coeff)
                elseif haskey(opt.affine_var_to_col, v)
                    push!(D_I, offset + 1); push!(D_J, opt.affine_var_to_col[v]); push!(D_V, coeff)
                end
            end
            target = s_con.value
            b_vec[offset + 1] = f_con.constant - target
        end
    end

    A = sparse(A_I, A_J, A_V, m, n)
    A_I = Int[]; A_J = Int[]; A_V = Float64[]
    D = sparse(D_I, D_J, D_V, m, p)
    D_I = Int[]; D_J = Int[]; D_V = Float64[]
    b = sparse(b_vec)
    b_vec = Float64[]

    # 4. Extract objective C, f, b0
    C_I = Int[]; C_V = Float64[]
    f_I = Int[]; f_V = Float64[]
    b0 = 0.0

    obj_type = MOI.get(opt.model, MOI.ObjectiveFunctionType())
    obj_f = MOI.get(opt.model, MOI.ObjectiveFunction{obj_type}())
    sense = MOI.get(opt.model, MOI.ObjectiveSense())

    if obj_f isa MOI.VariableIndex
        v = obj_f
        if haskey(opt.psd_var_to_col, v)
            push!(C_I, opt.psd_var_to_col[v]); push!(C_V, 1.0)
        elseif haskey(opt.affine_var_to_col, v)
            push!(f_I, opt.affine_var_to_col[v]); push!(f_V, 1.0)
        end
    elseif obj_f isa MOI.ScalarAffineFunction
        b0 = obj_f.constant
        for t in obj_f.terms
            v = t.variable
            if haskey(opt.psd_var_to_col, v)
                push!(C_I, opt.psd_var_to_col[v]); push!(C_V, t.coefficient)
            elseif haskey(opt.affine_var_to_col, v)
                push!(f_I, opt.affine_var_to_col[v]); push!(f_V, t.coefficient)
            end
        end
    end

    C = sparsevec(C_I, C_V, n)
    f_vec = sparsevec(f_I, f_V, p)

    sdp = SemidefiniteProgram(
        C = C,
        f = f_vec,
        b0 = b0,
        sense = sense,
        A = A,
        D = D,
        b = b,
        blocks = blocks,
        config = SanitizerConfig(
            verbose = opt.verbose,
            eliminate_free_variables = opt.eliminate_free_variables,
            eliminate_redundant_constraints = opt.eliminate_redundant_constraints,
            num_threads = opt.num_threads,
            cond_tol = opt.cond_tol
        )
    )
    A = spzeros(Float64, 0, 0)
    D = spzeros(Float64, 0, 0)
    b = spzeros(Float64, 0)

    if opt.verbose
        println("[SDPSanitizer.MOIWrapper] Before presolve: f = ", Vector(sdp.f))
        println("[SDPSanitizer.MOIWrapper] Before presolve: C nz = ", length(sdp.C.nzind))
        println("[SDPSanitizer.MOIWrapper] Before presolve: b0 = ", sdp.b0)
    end
    opt.sdp = sdp
    if opt.presolve
        try
            presolve!(sdp)
        catch e
            if e isa ErrorException && e.msg == "INFEASIBLE"
                opt.infeasible = true
                opt.infeasible_msg = "Presolve detected infeasibility in redundant constraints"
                opt.solve_time = time() - t_start
                opt.inner_solve_time = 0.0
                LAST_INNER_SOLVE_TIME[] = 0.0
                LAST_TOTAL_SOLVE_TIME[] = opt.solve_time
                if opt.verbose
                    println("[SDPSanitizer.MOIWrapper] $(opt.infeasible_msg)")
                end
                return
            else
                rethrow(e)
            end
        end
    end
    m_before_sieve = size(sdp.A, 1)
    if opt.detect_infeasibility
        try
            sieve!(sdp; only_detect_infeasibility = true)
        catch e
            if e isa ErrorException && e.msg == "INFEASIBLE"
                opt.infeasible = true
                opt.infeasible_msg = "Sieve infeasibility detection detected infeasibility"
                opt.solve_time = time() - t_start
                opt.inner_solve_time = 0.0
                LAST_INNER_SOLVE_TIME[] = 0.0
                LAST_TOTAL_SOLVE_TIME[] = opt.solve_time
                if opt.verbose
                    println("[SDPSanitizer.MOIWrapper] $(opt.infeasible_msg)")
                end
                return
            else
                rethrow(e)
            end
        end
    end

    if opt.facial_reduction
        try
            opt.sieve_active_con_idx = sieve!(sdp; only_detect_infeasibility = false)
            opt.facial_reduction_applied = true
        catch e
            if e isa ErrorException && e.msg == "INFEASIBLE"
                opt.infeasible = true
                opt.infeasible_msg = "Facial reduction detected infeasibility"
                opt.solve_time = time() - t_start
                opt.inner_solve_time = 0.0
                LAST_INNER_SOLVE_TIME[] = 0.0
                LAST_TOTAL_SOLVE_TIME[] = opt.solve_time
                if opt.verbose
                    println("[SDPSanitizer.MOIWrapper] $(opt.infeasible_msg)")
                end
                return
            else
                rethrow(e)
            end
        end
    end
    if opt.verbose
        println("[SDPSanitizer.MOIWrapper] After presolve: C nz = ", length(sdp.C.nzind))
        println("[SDPSanitizer.MOIWrapper] After presolve: b0 = ", sdp.b0)
    end

    # 5. Build reduced presolved MOI model
    presolved_model = MOI.Utilities.Model{Float64}()

    # Preserve PSD cones
    for (ci, dim) in opt.psd_con_indices
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        s = MOI.get(opt.model, MOI.ConstraintSet(), ci)
        if f isa MOI.VectorOfVariables
            new_vars = MOI.add_variables(presolved_model, length(f.variables))
            for (old_v, new_v) in zip(f.variables, new_vars)
                opt.inner_var_map[old_v] = new_v
            end
            new_ci = MOI.add_constraint(presolved_model, MOI.VectorOfVariables(new_vars), s)
            opt.inner_psd_con_map[ci] = new_ci
        end
    end

    # Preserve Nonnegatives
    for ci in opt.nonneg_con_indices
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        s = MOI.get(opt.model, MOI.ConstraintSet(), ci)
        if f isa MOI.VectorOfVariables
            new_vars = MOI.VariableIndex[]
            for old_v in f.variables
                if !haskey(opt.inner_var_map, old_v)
                    opt.inner_var_map[old_v] = MOI.add_variable(presolved_model)
                end
                push!(new_vars, opt.inner_var_map[old_v])
            end
            MOI.add_constraint(presolved_model, MOI.VectorOfVariables(new_vars), s)
        end
    end

    # Preserve GreaterThan bounds
    for ci in opt.gt_con_indices
        f = MOI.get(opt.model, MOI.ConstraintFunction(), ci)
        s = MOI.get(opt.model, MOI.ConstraintSet(), ci)
        old_v = f isa MOI.VariableIndex ? f : f.variable
        if !haskey(opt.inner_var_map, old_v)
            opt.inner_var_map[old_v] = MOI.add_variable(presolved_model)
        end
        new_v = opt.inner_var_map[old_v]
        MOI.add_constraint(presolved_model, new_v, s)
    end

    # Add reduced objective
    new_obj_terms = MOI.ScalarAffineTerm{Float64}[]
    for (col, val) in zip(sdp.C.nzind, sdp.C.nzval)
        v_old = opt.psd_vars[col]
        push!(new_obj_terms, MOI.ScalarAffineTerm(val, opt.inner_var_map[v_old]))
    end
    MOI.set(presolved_model, MOI.ObjectiveSense(), sense)
    MOI.set(presolved_model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(new_obj_terms, sdp.b0))

    # Add reduced affine equality constraints Ã(Z) + b̃ = 0
    m_new = size(sdp.A, 1)
    if m_new > 0
        nnz_A = nnz(sdp.A)
        con_terms = Vector{MOI.VectorAffineTerm{Float64}}(undef, nnz_A)
        rows_A = rowvals(sdp.A)
        vals_A = nonzeros(sdp.A)
        term_idx = 0
        max_coeff = 0.0
        for col in 1:size(sdp.A, 2)
            v_inner = opt.inner_var_map[opt.psd_vars[col]]
            for ptr in nzrange(sdp.A, col)
                term_idx += 1
                v_coeff = vals_A[ptr]
                abs_c = abs(v_coeff)
                if abs_c > max_coeff
                    max_coeff = abs_c
                end
                con_terms[term_idx] = MOI.VectorAffineTerm(rows_A[ptr], MOI.ScalarAffineTerm(v_coeff, v_inner))
            end
        end
        if max_coeff > 1e15
            @warn "[SDPSanitizer] Presolved conic matrix A contains huge coefficient: $max_coeff (max allowed before Mosek error 1482 is 1e20). Consider checking network conditioning."
        end
        con_constants = Vector(sdp.b)
        con_func = MOI.VectorAffineFunction(con_terms, con_constants)
        opt.inner_presolved_con = MOI.add_constraint(presolved_model, con_func, MOI.Zeros(m_new))
        con_terms = MOI.VectorAffineTerm{Float64}[]
        con_constants = Float64[]
        con_func = nothing
    end

    # 6. Copy presolved model to inner solver and optimize
    inner_map = MOI.copy_to(opt.inner, presolved_model)
    for (old_v, mid_v) in opt.inner_var_map
        opt.inner_var_map[old_v] = inner_map[mid_v]
    end
    if opt.inner_presolved_con !== nothing
        opt.inner_presolved_con = inner_map[opt.inner_presolved_con]
    end

    # Free intermediate presolved model and large conic matrices before inner solver runs
    MOI.empty!(presolved_model)
    MOI.empty!(opt.model)
    sdp.A = spzeros(Float64, 0, 0)
    sdp.D = spzeros(Float64, 0, 0)
    GC.gc()

    if opt.verbose
        println("[SDPSanitizer.MOIWrapper] Solving reduced problem with inner solver...")
    end
    t_inner_start = time()
    MOI.optimize!(opt.inner)
    opt.inner_solve_time = time() - t_inner_start
    LAST_INNER_SOLVE_TIME[] = opt.inner_solve_time

    # 7. Solution recovery
    status = MOI.get(opt.inner, MOI.PrimalStatus())
    has_results = try
        MOI.get(opt.inner, MOI.ResultCount()) >= 1
    catch
        false
    end
    if status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) || (has_results && status != MOI.NO_SOLUTION)
        try
            Z_sol = [MOI.get(opt.inner, MOI.VariablePrimal(), opt.inner_var_map[v]) for v in opt.psd_vars]
            opt.recovered_affine_primal = recover_affine_solution(sdp, Z_sol)
        catch e
            if opt.verbose
                println("[SDPSanitizer.MOIWrapper] Warning: Primal recovery failed ($status): $e")
            end
        end
    end

    dual_status = MOI.get(opt.inner, MOI.DualStatus())
    can_try_dual = dual_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) ||
                   (has_results && dual_status != MOI.NO_SOLUTION)
    if (can_try_dual || opt.inner_presolved_con === nothing) && !opt.facial_reduction_applied
        try
            y_presolved = if opt.inner_presolved_con !== nothing && can_try_dual
                MOI.get(opt.inner, MOI.ConstraintDual(), opt.inner_presolved_con)
            else
                Float64[]
            end
            if opt.inner_presolved_con === nothing || !isempty(y_presolved)
                opt.recovered_dual = recover_dual_solution(sdp, y_presolved)
            end
        catch e
            if opt.verbose
                println("[SDPSanitizer.MOIWrapper] Warning: Dual recovery failed ($dual_status): $e")
            end
        end
    end
    opt.solve_time = time() - t_start
    LAST_TOTAL_SOLVE_TIME[] = opt.solve_time
    if opt.verbose
        presolve_time = max(0.0, opt.solve_time - opt.inner_solve_time)
        println("[SDPSanitizer.MOIWrapper] SDP presolve time: ", round(presolve_time, digits=4), " s, SDP solve time: ", round(opt.inner_solve_time, digits=4), " s")
    end
end

# Attribute queries
MOI.get(opt::MOIWrapper, attr::MOI.TerminationStatus) = opt.infeasible ? MOI.INFEASIBLE : MOI.get(opt.inner, attr)
MOI.get(opt::MOIWrapper, attr::MOI.PrimalStatus) = opt.infeasible ? MOI.NO_SOLUTION : MOI.get(opt.inner, attr)
function MOI.get(opt::MOIWrapper, attr::MOI.DualStatus)
    if opt.infeasible || opt.facial_reduction_applied
        return MOI.NO_SOLUTION
    end
    inner_st = MOI.get(opt.inner, attr)
    if inner_st == MOI.NO_SOLUTION && opt.inner_presolved_con === nothing && MOI.get(opt.inner, MOI.PrimalStatus()) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
        return MOI.FEASIBLE_POINT
    end
    if isempty(opt.recovered_dual) && opt.presolve && ((opt.eliminate_free_variables && !isempty(opt.affine_vars)) || opt.eliminate_redundant_constraints)
        return MOI.NO_SOLUTION
    end
    return inner_st
end
MOI.get(opt::MOIWrapper, attr::MOI.ObjectiveValue) = opt.infeasible ? NaN : MOI.get(opt.inner, attr)
MOI.get(opt::MOIWrapper, attr::MOI.RawStatusString) = opt.infeasible ? opt.infeasible_msg : MOI.get(opt.inner, attr)
MOI.get(opt::MOIWrapper, attr::MOI.SolveTimeSec) = opt.solve_time > 0.0 ? opt.solve_time : MOI.get(opt.inner, attr)
MOI.get(opt::MOIWrapper, attr::MOI.Silent) = opt.silent

function MOI.get(opt::MOIWrapper, attr::MOI.VariablePrimal, v::MOI.VariableIndex)
    has_presolve_work = opt.presolve && ((opt.eliminate_free_variables && !isempty(opt.affine_vars)) || opt.eliminate_redundant_constraints)
    has_sieve_work = opt.detect_infeasibility || opt.facial_reduction
    if !has_presolve_work && !has_sieve_work
        return MOI.get(opt.inner, attr, opt.inner_var_map[v])
    end
    if haskey(opt.psd_var_to_col, v)
        return MOI.get(opt.inner, attr, opt.inner_var_map[v])
    elseif haskey(opt.affine_var_to_col, v)
        idx = opt.affine_var_to_col[v]
        if isempty(opt.recovered_affine_primal) || idx > length(opt.recovered_affine_primal)
            error("Affine variable primal not available in SDPSanitizer.MOIWrapper (PrimalStatus = $(MOI.get(opt.inner, MOI.PrimalStatus())))")
        end
        return opt.recovered_affine_primal[idx]
    else
        error("Variable $v not recognized in SDPSanitizer.MOIWrapper")
    end
end

function MOI.get(opt::MOIWrapper, attr::MOI.ConstraintDual, c::MOI.ConstraintIndex)
    if opt.facial_reduction_applied
        error("Constraint duals cannot be recovered when facial reduction is applied because the constraint space was altered.")
    end
    has_presolve_work = opt.presolve && ((opt.eliminate_free_variables && !isempty(opt.affine_vars)) || opt.eliminate_redundant_constraints)
    has_sieve_work = opt.detect_infeasibility || opt.facial_reduction
    if !has_presolve_work && !has_sieve_work
        inner_ci = get(opt.inner_affine_con_map, c, c)
        return MOI.get(opt.inner, attr, inner_ci)
    end
    offset = opt.con_row_offsets[c]
    idx = findfirst(==(c), opt.affine_con_indices)
    dim = opt.affine_con_dims[idx]
    if isempty(opt.recovered_dual) || offset + dim > length(opt.recovered_dual)
        error("Constraint duals not available in SDPSanitizer.MOIWrapper (inner DualStatus = $(MOI.get(opt.inner, MOI.DualStatus())), TerminationStatus = $(MOI.get(opt.inner, MOI.TerminationStatus())))")
    end
    slice = opt.recovered_dual[offset+1 : offset+dim]
    return dim == 1 && c isa MOI.ConstraintIndex{<:MOI.ScalarAffineFunction} ? slice[1] : slice
end

# Pass-through for any other attribute
MOI.get(opt::MOIWrapper, attr::MOI.AbstractOptimizerAttribute) = MOI.get(opt.inner, attr)
MOI.get(opt::MOIWrapper, attr::MOI.AbstractModelAttribute) = MOI.get(opt.inner, attr)
