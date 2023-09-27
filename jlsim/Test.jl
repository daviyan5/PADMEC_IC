import LinearAlgebra as LA
using MKL

include("Helpers.jl")
include("MeshHandler.jl")
include("TPFASolver.jl")

import .Helpers
import .MeshHandler
import .TPFASolver

import JLD2
import DelimitedFiles as DF
import TimerOutputs as TO
import BenchmarkTools as BT
import StatsBase as SB

using PyCall
using Profile
using PProf

@pyinclude("plots.py")

function linear_case(mesh :: MeshHandler.Mesh, verbose :: Bool = false) :: TPFASolver.Solver
    function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return x + y + z
    end
        
    function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return zeros(length(x))
    end
    function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
        return zeros(length(x))
    end
    permeability = Helpers.get_tensor(1., mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "x + y + z")
    xv, yv, zv, idx = Helpers.get_column.(mesh.volumes_centers, 1), 
                      Helpers.get_column.(mesh.volumes_centers, 2), 
                      Helpers.get_column.(mesh.volumes_centers, 3), 
                      Vector(1:mesh.nvols)
    args = Dict("mesh_filename" => "",
                "mesh"          => mesh,
                "dirichlet"     => dirichlet,
                "neumann"       => neumann,
                "source"        => source,
                "permeability"  => permeability,
                "analytical"    => dirichlet(xv, yv, zv, idx, mesh),
                "vtk"           => false)
    TPFASolver.solve_TPFA!(solver, args)
    if verbose == true
        Helpers.verbose("For $(solver.name):", "OUT")
        show(solver)
        println()
    end
    return solver
end

function quadratic_case(mesh :: MeshHandler.Mesh, verbose :: Bool = false) :: TPFASolver.Solver
    function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return x.^2 + y.^2 + z.^2
    end
        
    function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return zeros(length(x))
    end
    function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
        return ones(length(x)) .* -6
    end
    permeability = Helpers.get_tensor(1., mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "x^2 + y^2 + z^2")
    xv, yv, zv, idx = Helpers.get_column.(mesh.volumes_centers, 1), 
                      Helpers.get_column.(mesh.volumes_centers, 2), 
                      Helpers.get_column.(mesh.volumes_centers, 3), 
                      Vector(1:mesh.nvols)
    args = Dict("mesh_filename" => "",
                "mesh"          => mesh,
                "dirichlet"     => dirichlet,
                "neumann"       => neumann,
                "source"        => source,
                "permeability"  => permeability,
                "analytical"    => dirichlet(xv, yv, zv, idx, mesh),
                "vtk"           => false)
    TPFASolver.solve_TPFA!(solver, args)
    if verbose == true
        Helpers.verbose("For $(solver.name):", "OUT")
        show(solver)
        println()
    end
    return solver
end

function extra_case1(mesh :: MeshHandler.Mesh, verbose :: Bool = false) :: TPFASolver.Solver
    function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return sin.(x) + cos.(y) + exp.(z)
    end
        
    function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return zeros(length(x))
    end
    function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
        return sin.(x) + cos.(y) - exp.(z)
    end
    permeability = Helpers.get_tensor(1., mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "sin(x) + cos(y) + exp(z)")
    xv, yv, zv, idx = Helpers.get_column.(mesh.volumes_centers, 1), 
                      Helpers.get_column.(mesh.volumes_centers, 2), 
                      Helpers.get_column.(mesh.volumes_centers, 3), 
                      Vector(1:mesh.nvols)
    args = Dict("mesh_filename" => "",
                "mesh"          => mesh,
                "dirichlet"     => dirichlet,
                "neumann"       => neumann,
                "source"        => source,
                "permeability"  => permeability,
                "analytical"    => dirichlet(xv, yv, zv, idx, mesh),
                "vtk"           => false)
    TPFASolver.solve_TPFA!(solver, args)
    if verbose == true
        Helpers.verbose("For $(solver.name):", "OUT")
        show(solver)
        println()
    end
    return solver
end

function extra_case2(mesh :: MeshHandler.Mesh, verbose :: Bool = false) :: TPFASolver.Solver
    function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return (x .+ 1) .* log.(1 .+ x) + 1 ./ (y .+ 1) + z.^2
    end
        
    function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return zeros(length(x))
    end
    function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
        return -2 .- (1 ./ (1 .+ x)) - (2 ./ ((1 .+ y) .^ 3))
    end
    permeability = Helpers.get_tensor(1., mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "(x + 1) * log(1 + x) + 1/(y + 1) + z^2")
    xv, yv, zv, idx = Helpers.get_column.(mesh.volumes_centers, 1), 
                      Helpers.get_column.(mesh.volumes_centers, 2), 
                      Helpers.get_column.(mesh.volumes_centers, 3), 
                      Vector(1:mesh.nvols)
    args = Dict("mesh_filename" => "",
                "mesh"          => mesh,
                "dirichlet"     => dirichlet,
                "neumann"       => neumann,
                "source"        => source,
                "permeability"  => permeability,
                "analytical"    => dirichlet(xv, yv, zv, idx, mesh),
                "vtk"           => false)
    TPFASolver.solve_TPFA!(solver, args)
    if verbose == true
        Helpers.verbose("For $(solver.name):", "OUT")
        show(solver)
        println()
    end
    return solver
    
end

function qfive_spot(mesh :: MeshHandler.Mesh, box_dimensions :: Tuple = (1, 1, 1), verbose :: Bool = false) :: TPFASolver.Solver
    
    Lx, Ly, Lz = box_dimensions
    d1 = 10000.
    d2 = 1.
    k1 = 100.
    k2 = 100.
    vq = 0.

    function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        p1 = 0.1
        p2 = 0.1
        l1, c1 = Lx / 8, Ly / 8
        l2, c2 = Lx / 8, Ly / 8
        dist0 = (x .>= 0) .& (x .<= l1) .& (y .>= 0) .& (y .<= c1)
        dist1 = (x .>= Lx - l2) .& (x .<= Lx) .& (y .>= Ly - c2) .& (y .<= Ly)

        v1 = findall(dist0 .== true)
        v2 = findall(dist1 .== true)

        dp = Vector{Union{Float64, Nothing}}([nothing for i in 1:length(x)])
        dp[v1] .= [d1]
        dp[v2] .= [d2]
        return dp
    end
        
    function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
        return zeros(length(x))
    end
    function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
        return ones(length(x)) .* vq
    end
    function analytical(x :: Vector, y :: Vector, z :: Vector) :: Vector
        ref = DF.readdlm("vtks/qfive_spot_ref.txt", ',')
        xr  = ref[:, 1]
        yr  = ref[:, 2]
        zr  = ref[:, 3] 
        pr  = ref[:, 4]
        
        n   = Integer(round(sqrt(length(x))))
        dx  = Lx / n
        dy  = Ly / n
        i = ceil.(Int64, yr ./ dy)
        j = ceil.(Int64, xr ./ dx)
        idx = ((i .- 1) .* n) .+ j
        a_p = zeros(n * n)
        a_pv = @view a_p[idx]
        a_pv .+= pr
        m = length(idx) / length(unique(idx))
        return a_p ./ m

    end

    permeability = Helpers.get_tensor(k1, mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "1/4 de Five-Spot")
    xv, yv, zv, idx = Helpers.get_column.(mesh.volumes_centers, 1), 
                      Helpers.get_column.(mesh.volumes_centers, 2), 
                      Helpers.get_column.(mesh.volumes_centers, 3), 
                      Vector(1:mesh.nvols)

    args = Dict("mesh_filename" => "",
                "mesh"          => mesh,
                "dirichlet"     => dirichlet,
                "neumann"       => neumann,
                "source"        => source,
                "permeability"  => permeability,
                "analytical"    => analytical(xv, yv, zv),
                "vtk"           => false)

    TPFASolver.solve_TPFA!(solver, args)

    if verbose == true
        Helpers.verbose("For $(solver.name):", "OUT")
        show(solver)
        println()
    end
    return solver
    
end

function create_qfive_spot_ref()
    py"""
    
    import numpy as np  
    import numpy as np
    import os
    import sys
    from scipy.io import savemat
    def create_python_ref(x, y, z, p):
        d = {
            "x" : x,
            "y" : y,
            "z" : z,
            "p" : p
        }
        
        np.save("./vtks/qfive_spot_ref.npy", d)
        

    def create_matlab_ref(x, y, z, p):
        savemat("./vtks/qfive_spot_ref.mat", {"x": x, "y": y, "z": z, "p": p})
        
    """
    meshfile = "./mesh/box_9.msh"
    mesh :: Union{MeshHandler.Mesh, Nothing} = nothing
    if isfile("./mesh/mesh9.jld2")
        file = JLD2.jldopen("./mesh/mesh9.jld2", "r")
        mesh = file["mesh"]
    else
        mesh = MeshHandler.Mesh(meshfile)
        JLD2.jldsave("./mesh/mesh9.jld2"; mesh)
    end
    solver = qfive_spot(mesh, (6, 4, 1), true)
    xv, yv, zv =    Helpers.get_column.(mesh.volumes_centers, 1), 
                    Helpers.get_column.(mesh.volumes_centers, 2), 
                    Helpers.get_column.(mesh.volumes_centers, 3)

    DF.writedlm("./vtks/qfive_spot_ref.txt", [xv yv zv solver.p_TPFA], ',')
    py"create_python_ref"(xv, yv, zv, solver.p_TPFA)
    py"create_matlab_ref"(xv, yv, zz, solver.p_TPFA)
end
function run_tests(meshfiles :: Array, meshfiles_qfive_spot :: Array, nrepeats :: Int64 = 1, try_precompile :: Bool = true, test_case :: Int64 = 1)
    if try_precompile
        meshf = "./mesh/cube_hex.msh"
        meshqfive = "./mesh/box_2.msh"
        Helpers.verbose("Precompiling with two dummy cases...", "INFO")
        linear_case(MeshHandler.Mesh(meshf))
        qfive_spot(MeshHandler.Mesh(meshqfive), (6, 4, 1))
    end
    p1 = Helpers.Problem("linear_case", linear_case, meshfiles)
    p2 = Helpers.Problem("quadratic_case", quadratic_case, meshfiles)
    p3 = Helpers.Problem("extra_case1", extra_case1, meshfiles)
    p4 = Helpers.Problem("extra_case2", extra_case2, meshfiles)
    p5 = Helpers.Problem("qfive_spot", qfive_spot, meshfiles_qfive_spot)
    problems = [p1, p2, p3, p4, p5]
    problems = [problems[test_case]]
    Helpers.verbose("Loading meshes and mesh_params", "OUT")    
    meshes_dict = Dict()
    for problem in problems
        for meshfile in problem.meshfiles
            if haskey(meshes_dict, meshfile)
                continue
            end
            Helpers.verbose("Loading $(meshfile)...", "INFO")
            mesh = MeshHandler.Mesh(meshfile)
            meshes_dict[meshfile] = mesh
            Helpers.verbose("Finished loading $(meshfile) : $(mesh.nvols) volumes", "INFO")
        end
    end
    
    verbose = false
    Helpers.verbose("Starting tests", "OUT")
    summaryIO = open("saida.txt", "w")
    for i in 1:length(problems)
        k = 1
        solver_name = ""
        for meshfile in problems[i].meshfiles
            mesh = meshes_dict[meshfile]
            for j in 1:nrepeats
                memuse = Helpers.memuse()
                Helpers.verbose("Starting [$(j)/$(nrepeats)] rep of [$(k)/$(length(problems[i].meshfiles))] in $(problems[i].name)", "OUT")
                Helpers.verbose("Memory use: $(memuse)", "INFO")
                if memuse > 1.0e4
                    counter = 0 
                    Helpers.verbose("Memory use too high! -> $(Helpers.memuse())", "INFO")
                    while Helpers.memuse() > 1.0e4
                        #GC.gc()
                        counter += 1
                        if counter == 3
                            break
                        end
                    end
                end
                solver :: TPFASolver.Solver = TPFASolver.Solver()
                if problems[i].name != "qfive_spot"
                    solver = problems[i].handle(mesh, verbose)
                else
                    solver = problems[i].handle(mesh, (6, 4, 1), verbose)
                end
                solver_name = string(solver.name)
                Helpers.verbose("Finished [$(j)/$(nrepeats)] reps of [$(k)/$(length(problems[i].meshfiles))] in $(solver.name)!", "INFO")
                Helpers.add_to_problem!(problems[i], k, j, solver.mesh.nvols, solver.error, solver.times, solver.memory)
            end
            k += 1
            
        end
        pr          = problems[i]
        name        = solver_name
        meshfiles   = pr.meshfiles
        nvols       = pr.nvols_arr
        error       = pr.avg_error_arr
        times       = pr.avg_time_arr
        memory      = pr.avg_memory_arr

        d = Dict("name"         => name,
                 "meshfiles"    => meshfiles,
                 "nvols"        => nvols,
                 "error"        => error,
                 "times"        => times,
                 "memory"       => memory)

        JLD2.jldsave("./results/$(pr.name).jld2"; d)
    end

end


function main()
    
    #create_qfive_spot_ref()
    test_case = 0
    for X in ARGS
        test_case = parse(Int64, X)
    end
    meshfiles_qfive_spot = ["./mesh/box_$(i).msh" for i in 2:7]
    meshfiles           = ["./mesh/box_$(i)acc.msh" for i in 0:14]
    nrepeats = 5
    #run_tests(meshfiles, meshfiles_qfive_spot, nrepeats, true, test_case)
    case1 = JLD2.jldopen("./results/linear_case.jld2", "r")["d"]
    case2 = JLD2.jldopen("./results/quadratic_case.jld2", "r")["d"]
    case3 = JLD2.jldopen("./results/extra_case1.jld2", "r")["d"]
    case4 = JLD2.jldopen("./results/extra_case2.jld2", "r")["d"]
    case5 = JLD2.jldopen("./results/qfive_spot.jld2", "r")["d"]
    cases = [case1, case2, case4, case5]
    py"plot_times"(cases)
    py"plot_memory"(cases)
    py"plot_accurracy"(cases)
end
if !isinteractive()
    main()
end