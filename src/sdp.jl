"""
    SanitizerConfig(; verbose::Bool = false)

Configuration options for presolving SDPs in `SDPSanitizer`.

# Fields
- `verbose::Bool`: Print logging and timing diagnostics during presolve (default: `false`).
"""
Base.@kwdef mutable struct SanitizerConfig
    verbose::Bool = false
end

"""
    AffineRecoveryInfo

Internal recovery data stored in `SemidefiniteProgram` to reconstruct eliminated affine variables `z`.

# Fields
- `F_DB::LinearAlgebra.Factorization{Float64}`: LU factorization of the basis block `D_B`.
- `b_B::Vector{Float64}`: Basis slice of the left-hand-side constraint offset `b`.
- `A_B::SparseMatrixCSC{Float64, Int}`: Basis slice of constraint matrix `A`.
- `independent_cols::Vector{Int}`: Indices of linearly independent columns in the original `D`.
- `p_orig::Int`: Original number of affine variables.
"""
struct AffineRecoveryInfo
    F_DB::LinearAlgebra.Factorization{Float64}
    b_B::Vector{Float64}
    A_B::SparseMatrixCSC{Float64, Int}
    independent_cols::Vector{Int}
    p_orig::Int
end

"""
    DualRecoveryInfo

Internal recovery data stored in `SemidefiniteProgram` to reconstruct eliminated equality constraint duals `y`.

# Fields
- `B::Vector{Int}`: Basis row indices of the affine constraint matrix `D`.
- `N::Vector{Int}`: Non-basis row indices of the affine constraint matrix `D`.
- `F_DB::LinearAlgebra.Factorization{Float64}`: LU factorization of the basis block `D_B`.
- `f_indep::Vector{Float64}`: Objective coefficients of independent affine variables.
- `D_N::SparseMatrixCSC{Float64, Int}`: Non-basis row slice of `D`.
- `m::Int`: Total original number of constraints.
"""
struct DualRecoveryInfo
    B::Vector{Int}
    N::Vector{Int}
    F_DB::LinearAlgebra.Factorization{Float64}
    f_indep::Vector{Float64}
    D_N::SparseMatrixCSC{Float64, Int}
    m::Int
end

"""
    SemidefiniteProgram(; C, f, b0, A, D, b, config, recovery_info, dual_recovery_info)

Represents a semidefinite program with conic and affine variables:

    min/max  ⟨C, Z⟩ + fᵀ z + b₀
    s.t.     A(Z) + D z + b = 0
             Z ⪰ 0

# Symmetric Matrix Representation (Column-Major Flattened Upper Triangular)
Symmetric matrices (such as `Z`, `C`, and the rows of `A`) are represented as flattened
sparse vectors respecting column-major order of the upper triangle.
For an `n × n` symmetric matrix, entry `(i, j)` with `1 ≤ i ≤ j ≤ n` maps to 1-based index:

    k = i + div(j * (j - 1), 2)

yielding total length `N_triu = div(n * (n + 1), 2)`.
Off-diagonal entries (`i < j`) are scaled by `√2` so that the standard inner product
`⟨C, Z⟩ = tr(C Z)` corresponds to the Euclidean dot product `Cᵀ Z`, matching
`MathOptInterface.ScaledPositiveSemidefiniteConeTriangle(n)`.

# Fields
- `C::SparseVector{Float64, Int}`: Flattened upper triangular objective coefficients for `Z` (length `N_triu`).
- `f::SparseVector{Float64, Int}`: Objective coefficients for free affine variables `z` (length `p`).
- `b0::Float64`: Scalar objective offset.
- `sense::MOI.OptimizationSense`: Objective optimization sense (`MOI.MIN_SENSE` or `MOI.MAX_SENSE`, default: `MOI.MIN_SENSE`).
- `A::SparseMatrixCSC{Float64, Int}`: Conic equality constraint matrix of size `m × N_triu`.
- `D::SparseMatrixCSC{Float64, Int}`: Affine equality constraint matrix of size `m × p`.
- `b::SparseVector{Float64, Int}`: Left-hand-side constraint constant vector of length `m` (i.e. `A(Z) + D z + b = 0`).
- `config::SanitizerConfig`: Presolve configuration.
- `recovery_info::Union{AffineRecoveryInfo, Nothing}`: Presolve state used by `recover_affine_solution`.
- `dual_recovery_info::Union{DualRecoveryInfo, Nothing}`: Presolve state used by `recover_dual_solution`.
"""
Base.@kwdef mutable struct SemidefiniteProgram
    C::SparseVector{Float64, Int} = spzeros(Float64, 0)
    f::SparseVector{Float64, Int} = spzeros(Float64, 0)
    b0::Float64 = 0.0
    sense::MOI.OptimizationSense = MOI.MIN_SENSE
    A::SparseMatrixCSC{Float64, Int} = spzeros(Float64, 0, 0)
    D::SparseMatrixCSC{Float64, Int} = spzeros(Float64, 0, 0)
    b::SparseVector{Float64, Int} = spzeros(Float64, 0)
    blocks::Vector{Int} = Int[]

    config::SanitizerConfig = SanitizerConfig()
    recovery_info::Union{AffineRecoveryInfo, Nothing} = nothing
    dual_recovery_info::Union{DualRecoveryInfo, Nothing} = nothing
end

Base.copy(cfg::SanitizerConfig) = SanitizerConfig(verbose = cfg.verbose)

function Base.copy(sdp::SemidefiniteProgram)
    return SemidefiniteProgram(
        C = copy(sdp.C),
        f = copy(sdp.f),
        b0 = sdp.b0,
        sense = sdp.sense,
        A = copy(sdp.A),
        D = copy(sdp.D),
        b = copy(sdp.b),
        blocks = copy(sdp.blocks),
        config = copy(sdp.config),
        recovery_info = sdp.recovery_info,
        dual_recovery_info = sdp.dual_recovery_info
    )
end

"""
    as_model(sdp::SemidefiniteProgram; blocks::Union{Nothing, Vector{Int}} = nothing)::MOI.ModelLike

Translate `sdp` into a `MathOptInterface.ModelLike` instance.

Conic variables `Z` are constrained block-by-block according to `blocks` (or `sdp.blocks`),
preserving individual blocks via `MOI.ScaledPositiveSemidefiniteConeTriangle(b)`.
If no block information is provided, infers a single block if possible, or treats
variables as individual 1x1 scalar blocks.
Affine variables `z` are unconstrained, the objective sense is set to `sdp.sense`,
and constraints `A(Z) + D z + b = 0` are added as `A(Z) + D z == -b`.
"""
function as_model(sdp::SemidefiniteProgram; blocks::Union{Nothing, Vector{Int}} = nothing)::MOI.ModelLike
    N_triu = size(sdp.A, 2)
    m = size(sdp.A, 1)
    p = size(sdp.D, 2)

    eff_blocks = if blocks !== nothing && !isempty(blocks)
        blocks
    elseif !isempty(sdp.blocks)
        sdp.blocks
    else
        det = 8 * N_triu + 1
        s = round(Int, sqrt(det))
        if s * s == det && isodd(s)
            [div(s - 1, 2)]
        else
            fill(1, N_triu)
        end
    end

    model = MOI.Utilities.Model{Float64}()

    # 1. Variables: Z in ScaledPositiveSemidefiniteConeTriangle per block, z unconstrained
    Z_vars = MOI.VariableIndex[]
    for b in eff_blocks
        if b > 0
            vars, _ = MOI.add_constrained_variables(model, MOI.ScaledPositiveSemidefiniteConeTriangle(b))
            append!(Z_vars, vars)
        end
    end
    @assert length(Z_vars) == N_triu "Sum of block triangular dimensions ($(length(Z_vars))) does not match columns of A ($N_triu)"
    z_vars = MOI.add_variables(model, p)

    # 2. Objective: ⟨C, Z⟩ + fᵀ z + b₀
    obj_terms = MOI.ScalarAffineTerm{Float64}[]

    for (k, v) in zip(sdp.C.nzind, sdp.C.nzval)
        push!(obj_terms, MOI.ScalarAffineTerm(v, Z_vars[k]))
    end
    for (k, v) in zip(sdp.f.nzind, sdp.f.nzval)
        push!(obj_terms, MOI.ScalarAffineTerm(v, z_vars[k]))
    end

    MOI.set(model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(obj_terms, sdp.b0))
    MOI.set(model, MOI.ObjectiveSense(), sdp.sense)

    # 3. Constraints: A(Z) + D z + b = 0  <=>  A(Z) + D z == -b
    constraint_terms = [MOI.ScalarAffineTerm{Float64}[] for _ in 1:m]

    # Fast iteration over CSC sparse columns for A
    rows_A = rowvals(sdp.A)
    vals_A = nonzeros(sdp.A)
    for k in 1:N_triu
        for ptr in nzrange(sdp.A, k)
            row = rows_A[ptr]
            push!(constraint_terms[row], MOI.ScalarAffineTerm(vals_A[ptr], Z_vars[k]))
        end
    end

    # Fast iteration over CSC sparse columns for D
    rows_D = rowvals(sdp.D)
    vals_D = nonzeros(sdp.D)
    for k in 1:p
        for ptr in nzrange(sdp.D, k)
            row = rows_D[ptr]
            push!(constraint_terms[row], MOI.ScalarAffineTerm(vals_D[ptr], z_vars[k]))
        end
    end

    # Add constraints row by row
    for row in 1:m
        func = MOI.ScalarAffineFunction(constraint_terms[row], 0.0)
        MOI.add_constraint(model, func, MOI.EqualTo(-sdp.b[row]))
    end

    return model
end
