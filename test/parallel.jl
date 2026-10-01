using SDPSanitizer
using Test
using LinearAlgebra
using SparseArrays
using Random
import MathOptInterface as MOI
using Clarabel
using CSDP

@testset "Multi-threaded presolve equivalence and scaling" begin
    # Generate a medium-sized synthetic SDP with coupled free variables
    blocks = [8, 6, 4]
    p = 20
    m = 50
    rank_D = 12

    sdp_base, Z_true, z_true = synthetic_sdp(blocks, p, m, rank_D)

    # 1. Single-threaded reference (num_threads = 1)
    sdp_ref = copy(sdp_base)
    sdp_ref.config.num_threads = 1
    sdp_ref.config.verbose = false
    presolve!(sdp_ref)

    # 2. Test multi-threaded presolve with thread counts 2, 4, 8, and auto (0)
    for n_threads in [2, 4, 8, 0]
        sdp_par = copy(sdp_base)
        sdp_par.config.num_threads = n_threads
        sdp_par.config.verbose = false
        presolve!(sdp_par)

        # Check structural equality
        @test size(sdp_par.A) == size(sdp_ref.A)
        @test size(sdp_par.D) == size(sdp_ref.D)
        @test length(sdp_par.b) == length(sdp_ref.b)
        @test length(sdp_par.C) == length(sdp_ref.C)

        # Check numerical equality
        @test isapprox(sdp_par.b0, sdp_ref.b0, atol=1e-12)
        @test isapprox(Vector(sdp_par.b), Vector(sdp_ref.b), atol=1e-12)
        @test isapprox(Vector(sdp_par.C), Vector(sdp_ref.C), atol=1e-12)
        @test isapprox(Matrix(sdp_par.A), Matrix(sdp_ref.A), atol=1e-12)

        # Solve with CSDP and compare optimal value
        model_par = as_model(sdp_par)
        opt_par = MOI.instantiate(CSDP.Optimizer, with_bridge_type=Float64)
        MOI.set(opt_par, MOI.Silent(), true)
        idx_map_par = MOI.copy_to(opt_par, model_par)
        MOI.optimize!(opt_par)
        status = MOI.get(opt_par, MOI.TerminationStatus())
        @test status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        tol = status == MOI.OPTIMAL ? 1e-5 : 1e-2
        obj_par = MOI.get(opt_par, MOI.ObjectiveValue())
        @test isapprox(obj_par, 0.0, atol=tol)

        # Test affine solution recovery
        Z_vars = MOI.get(model_par, MOI.ListOfVariableIndices())
        Z_sol = MOI.get(opt_par, MOI.VariablePrimal(), [idx_map_par[v] for v in Z_vars])
        z_rec = recover_affine_solution(sdp_par, Z_sol)
        @test isapprox(sdp_base.A * Z_sol + sdp_base.D * z_rec + sdp_base.b, zeros(m), atol=tol)
    end
end

@testset "MOIWrapper thread configuration pass-through" begin
    # Test setting threads via constructor
    wrap1 = MOIWrapper(Clarabel.Optimizer; num_threads=4)
    @test wrap1.num_threads == 4

    # Test setting threads via RawOptimizerAttribute
    wrap2 = MOIWrapper(Clarabel.Optimizer)
    MOI.set(wrap2, MOI.RawOptimizerAttribute("threads"), 8)
    @test wrap2.num_threads == 8

    MOI.set(wrap2, MOI.RawOptimizerAttribute("num_threads"), 2)
    @test wrap2.num_threads == 2

    # Test setting detect_infeasibility and facial_reduction attributes
    MOI.set(wrap2, MOI.RawOptimizerAttribute("detect_infeasibility"), true)
    @test wrap2.detect_infeasibility == true

    MOI.set(wrap2, MOI.RawOptimizerAttribute("facial_reduction"), true)
    @test wrap2.facial_reduction == true
end
