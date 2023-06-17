include("Helpers.jl")
include("MeshHandler.jl")
include("TPFASolver.jl")

import .Helpers
import .MeshHandler
import .TPFASolver

function dirichlet(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
    return x + y + z
end
    
function neumann(x :: Vector, y :: Vector, z :: Vector, idx :: Vector, mesh :: MeshHandler.Mesh) :: Vector
    return zeros(length(x))
end
function source(x :: Vector, y :: Vector, z :: Vector, K :: Vector) :: Vector
    return zeros(length(x))
end
function main()
    
    mesh_filename = "./mesh/box_12acc.msh"
    mesh = MeshHandler.Mesh(mesh_filename)
    permeability = Helpers.get_tensor(1, mesh.nvols, 3, 3, true)
    solver = TPFASolver.Solver(true, true, "Test")
    args = Dict("mesh_filename" => mesh_filename,
                "mesh" => mesh,
                "dirichlet" => dirichlet,
                "neumann" => neumann,
                "source" => source,
                "permeability" => permeability)
    TPFASolver.solve_TPFA!(solver, args)
    show(solver)
    show(solver.bench_info)
    println()
    solver = TPFASolver.Solver(true, false, "Test2")
    TPFASolver.solve_TPFA!(solver, args)
    show(solver)
    show(solver.bench_info)
    println()
    return solver
end

s = main()