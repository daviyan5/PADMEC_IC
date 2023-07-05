include("Helpers.jl")
include("MeshHandler.jl")
include("TPFASolver.jl")

import .Helpers
import .MeshHandler
import .TPFASolver

import JLD2
import DelimitedFiles as DF
import LinearAlgebra as LA
import TimerOutputs as TO
using PyCall
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
                "vtk" => false)
    TPFASolver.solve_TPFA!(solver, args)
    TPFASolver.solve_TPFA!(solver, args)
    Helpers.verbose("For $(solver.name):", "OUT")
    show(solver)
    println()
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
                "vtk" => false)
    TPFASolver.solve_TPFA!(solver, args)
    TPFASolver.solve_TPFA!(solver, args)
    Helpers.verbose("For $(solver.name):", "OUT")
    show(solver)
    println()
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
                "vtk" => true)
    TPFASolver.solve_TPFA!(solver, args)
    TPFASolver.solve_TPFA!(solver, args)
    Helpers.verbose("For $(solver.name):", "OUT")
    show(solver)
    println()
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
                "vtk" => false)
    TPFASolver.solve_TPFA!(solver, args)
    TPFASolver.solve_TPFA!(solver, args)
    Helpers.verbose("For $(solver.name):", "OUT")
    show(solver)
    println()
    return solver
    
end

function Well2_case(mesh :: MeshHandler.Mesh, box_dimensions :: Tuple = (1, 1, 1), verbose :: Bool = false) :: TPFASolver.Solver
    
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
        ref = DF.readdlm("vtks/2WellRef.txt", ',')
        xr  = ref[:, 1]
        yr  = ref[:, 2]
        pr  = ref[:, 4]
        n   = Integer(sqrt(length(x)))
        dx  = Lx / n
        dy  = Ly / n
        i = ceil.(Int64, yr ./ dy)
        j = ceil.(Int64, xr ./ dx)
        idx = (i .- 1) .* n .+ j
        a_p = zeros(n * n)
        freq = zeros(n * n)
        for i in 1:size(idx)[1]
            a_p[idx[i]] += pr[i]
            freq[idx[i]] += 1
        end
        m = freq[1]
        return a_p ./ m

    end

    permeability = Helpers.get_tensor(k1, mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(verbose, true, "2Well")
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

    Helpers.verbose("For $(solver.name):", "OUT")
    show(solver)
    println()
    return solver
    
end

function run_tests(meshfiles :: Array, meshfiles_2Well :: Array, try_precompile :: Bool = true)
    if try_precompile
        meshfile = "./mesh/cube_hex.msh"
        aux_solver = linear_case(MeshHandler.Mesh(meshfile))
    end
    function general_test(meshfiles :: Array, test_functions :: Array{Function}, add_to_solver :: Function, 
                          n_tests :: Int64, n_reps :: Int64, res :: Dict, vols :: Array, verbose :: Bool = false, type = "GNR",
                          meshfiles_2Well :: Array = [], vols2Well :: Array = [])
        for (i, (meshfile, box_dimensions)) in enumerate(meshfiles)
            if i > n_tests
                break
            end
            mesh = MeshHandler.Mesh(meshfile)
            mesh2Well = i <= length(meshfiles_2Well) ? MeshHandler.Mesh(meshfiles_2Well[i][1]) : nothing
            if i <= length(meshfiles_2Well)
                append!(vols2Well, mesh2Well.nvols)
            end
            append!(vols, mesh.nvols)
            
            for j in 1:n_reps
                Helpers.verbose("Starting $(type) test [$(i)/$(n_tests), $(j)/$(n_reps)] for $(mesh.nvols)", "OUT")
                for f in test_functions
                    if f == Well2_case && mesh2Well !== nothing
                        solver = f(mesh2Well, meshfiles_2Well[i][2])
                        add_to_solver(solver, i, j, res)
                    elseif f != Well2_case
                        solver = f(mesh)
                        add_to_solver(solver, i, j, res)
                    end
                    
                end
                GC.gc()
            end
        end
    end
    function run_accuracy(meshfiles :: Array, n_tests :: Int64, n_reps :: Int64)
        n_tests = min(n_tests, length(meshfiles))
        println("Running accuracy tests")
        n_mesh = length(meshfiles)
        acc = Dict{String, Matrix{Float64}}()
        test_functions = [linear_case, quadratic_case, Well2_case, extra_case2]
        vols = []
        vols2Well = []
        function add_to_solver(solver :: TPFASolver.Solver, i :: Int64, j :: Int64, res :: Dict)
            name = solver.name
            if haskey(res, name)
                res[name][i, j] = solver.error
            else
                res[name] = zeros(Float64, n_tests, n_reps)
                res[name][i, j] = solver.error
            end
        end
        general_test(meshfiles, test_functions, add_to_solver, n_tests, n_reps, acc, vols, false, "ACC", meshfiles_2Well, vols2Well)
        Helpers.verbose("Done with accuracy tests", "OUT")
        Helpers.verbose("=", "OUT")
        for (name, matrix) in acc
            println("For $(name):")
            display(matrix)
            println()
        end
        py"plot_accuracy"(acc, vols, vols2Well)
    end

    function run_times(meshfiles :: Array, n_tests :: Int64, n_reps :: Int64)
        n_tests = min(n_tests, length(meshfiles))
        println("Running time tests")
        n_mesh = length(meshfiles)

        times = Dict{String, Dict{String, Matrix{Float64}}}()
        vols = []
        vols2Well = []
        test_functions = [linear_case, quadratic_case, Well2_case]

        function add_to_solver(solver :: TPFASolver.Solver, i :: Int64, j :: Int64, res :: Dict)
            name = solver.name
            if haskey(res, name)
                for key in TPFASolver.get_times_keys()
                    n_key = key[6:end]
                    if n_key == "Boundary Conditions"
                        n_key = "System Preparation"
                    end
                    res[name][n_key][i, j] += TO.time(solver.bench_info[key]) / 1e9
                end
                res[name]["Total Time"][i, j] += TO.tottime(solver.bench_info) / 1e9
            else
                res[name] = Dict{String, Matrix{Float64}}()
                for key in TPFASolver.get_times_keys()
                    n_key = key[6:end]
                    if n_key == "Boundary Conditions"
                        n_key = "System Preparation"
                    end
                    res[name][n_key] = zeros(Float64, n_tests, n_reps)
                    res[name][n_key][i, j] += TO.time(solver.bench_info[key]) / 1e9
                end
                res[name]["Total Time"] = zeros(Float64, n_tests, n_reps)
                res[name]["Total Time"][i, j] += TO.tottime(solver.bench_info) / 1e9
            end
        end
        general_test(meshfiles, test_functions, add_to_solver, n_tests, n_reps, times, vols,  false, "TIME", meshfiles_2Well, vols2Well)
        Helpers.verbose("Done with times tests", "OUT")
        Helpers.verbose("=", "OUT")
        for (name, dict) in times
            println("For $(name):")
            for (key, matrix) in dict
                println("For $(key):")
                display(matrix)
            end
            println()
        end
        py"plot_times"(times, vols, vols2Well)
        
    end

    function run_memory(meshfiles :: Array, n_tests :: Int64, n_reps :: Int64)
        n_tests = min(n_tests, length(meshfiles))
        println("Running time tests")
        n_mesh = length(meshfiles)

        mem = Dict{String, Matrix{Float64}}()
        vols = []
        vols2Well = []
        test_functions = [linear_case, quadratic_case, Well2_case]
        function add_to_solver(solver :: TPFASolver.Solver, i :: Int64, j :: Int64, res :: Dict)
            name = solver.name
            if haskey(res, name)
                res[name][i, j] += TO.totallocated(solver.bench_info) / 1e6
            else
                res[name] = zeros(Float64, n_tests, n_reps)
                res[name][i, j] += TO.totallocated(solver.bench_info) / 1e6 
            end
        end
        general_test(meshfiles, test_functions, add_to_solver, n_tests, n_reps, mem, vols, false, "MEM", meshfiles_2Well, vols2Well)
        Helpers.verbose("Done with memory tests", "OUT")
        Helpers.verbose("=", "OUT")
        for (name, matrix) in mem
            println("For $(name):")
            display(matrix)
            println()
        end
        py"plot_memory"(mem, vols, vols2Well)
    end
        
    n_tests, n_reps = 14, 1
    run_accuracy(meshfiles, n_tests, n_reps)

    n_tests, n_reps = 14, 10
    run_times(meshfiles,n_tests, n_reps)
    
    n_tests, n_reps = 14, 10
    run_memory(meshfiles, n_tests, n_reps)

end

function create_2Well_ref()
    meshfile = "./mesh/box_9.msh"
    mesh :: Union{MeshHandler.Mesh, Nothing} = nothing
    if isfile("./mesh/mesh9.jld2")
        file = JLD2.jldopen("./mesh/mesh9.jld2", "r")
        mesh = file["mesh"]
    else
        mesh = MeshHandler.Mesh(meshfile)
        JLD2.jldsave("./mesh/mesh9.jld2"; mesh)
    end
    solver = Well2_case(mesh, (6, 4, 1), true)
    xv, yv, zv =    Helpers.get_column.(mesh.volumes_centers, 1), 
                    Helpers.get_column.(mesh.volumes_centers, 2), 
                    Helpers.get_column.(mesh.volumes_centers, 3)

    DF.writedlm("./vtks/2WellRef.txt", [xv yv zv solver.p_TPFA], ',')
end
function main()
    
    create_2Well_ref()
    meshfiles_2Well = [("./mesh/box_$(i).msh", (6, 4, 1)) for i in 2:9]
    meshfiles       = [("./mesh/box_$(i)acc.msh", (1, 1, 1)) for i in 0:14]
    run_tests(meshfiles, meshfiles_2Well, true)
    #solver = extra_case1(MeshHandler.Mesh(meshfiles[2][1]), true)
    #show(solver.bench_info)
    #Well2_case(MeshHandler.Mesh(meshfiles[7][1]), meshfiles[7][2], true)
    
end

main()