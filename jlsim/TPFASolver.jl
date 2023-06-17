module TPFASolver

# include("Helpers.jl")     # Only if called directly
# include("MeshHandler.jl") # Only if called directly

import ..MeshHandler
import ..Helpers

import LinearAlgebra as LA
import TimerOutputs as TO

using PyCall, StaticArrays, SparseArrays


const StringNull        = Union{String, Nothing}
const BoolNull          = Union{Bool, Nothing}
const DictNull          = Union{Dict, Nothing}
const TupleNull         = Union{Tuple, Nothing}

const IntVectorNull     = Union{Vector{Int64}, Nothing}
const FloatVectorNull   = Union{Vector{Float64}, Nothing}
const SparseMatrixNull  = Union{SparseMatrixCSC{Float64, Int64}, Nothing}

const MeshNull          = Union{MeshHandler.Mesh, Nothing}
const PyObjectNull      = Union{PyObject, Nothing}
const TimerOutputNull   = Union{TO.TimerOutput, Nothing}


mutable struct Solver 
    verbose         :: Bool                                             # Verbose mode
    check           :: Bool                                             # Flag for checking if solution is valid
    name            :: String                                           # Name of the solver
    times           :: DictNull                                         # Dict for storing times
    mesh            :: MeshNull                                         # Mesh object

    permeability    :: Union{Vector{SMatrix{3, 3, Float64}}, Nothing}   # Permeability tensor flattened
    faces_normals   :: Union{Vector{SVector{3, Float64}}, Nothing}      # Coordinates of normal vectors internal faces-wise
    faces_trans     :: FloatVectorNull                                  # Transmissibility of each face

    h_L             :: FloatVectorNull                                  # Distance from face_centers to left volume
    h_R             :: FloatVectorNull                                  # Distance from face_centers to right volume

    At_TPFA         :: SparseMatrixNull                                 # Transmissibility matrix
    bt_TPFA         :: FloatVectorNull                                  # Source term
    p_TPFA          :: FloatVectorNull                                  # Pressure solution
    
    irel            :: FloatVectorNull                                  # Relative error element-wise
    numerical_p     :: FloatVectorNull                                  # Numerical pressure solution
    analytical_p    :: FloatVectorNull                                  # Analytical pressure solution
    
    faces_with_bc   :: IntVectorNull                                    # Faces with boundary conditions
    volumes_with_bc :: IntVectorNull                                    # Volumes with boundary conditions
    
    d_faces         :: IntVectorNull                                    # Faces with Dirichlet boundary conditions
    d_values        :: FloatVectorNull                                  # Values of Dirichlet boundary conditions
    d_volumes       :: IntVectorNull                                    # Volumes with Dirichlet boundary conditions

    n_faces         :: IntVectorNull                                    # Faces with Neumann boundary conditions
    n_values        :: FloatVectorNull                                  # Values of Neumann boundary conditions
    n_volume        :: IntVectorNull                                    # Volumes with Neumann boundary conditions

    bench_info      :: TimerOutputNull                                  # Object with times and memories
    vtk_filename    :: StringNull                                       # Name of the vtk file to be written

    function Solver(verbose::Bool = true, check::Bool = true, name::String = "MESH")
        new(verbose, check, name, nothing, nothing, # verbose, check, name, times, mesh,
            nothing, nothing, nothing,              # permeability, faces_normals, faces_trans
            nothing, nothing,                       # h_L, h_R
            nothing, nothing, nothing,              # At_TPFA, bt_TPFA, p_TPFA
            nothing, nothing, nothing,              # irel, numerical_p, analytical_p
            nothing, nothing,                       # faces_with_bc, volumes_with_bc
            nothing, nothing, nothing,              # d_faces, d_values, d_volumes
            nothing, nothing, nothing,              # n_faces, n_values, n_volume
            nothing, nothing)                       # bench_info
    end

end

function Base.show(io :: IO, m :: Solver)
    println(io, "Solver: '$(m.name)' - Verbose Mode: $(m.verbose) - Check Solution: $(m.check)")
    println(io, m.p_TPFA === nothing ? "No solution available." : "Solution available.")
    println("Mesh:")
    show(io, m.mesh)
    println()
end

function solve_TPFA!(solver :: Solver, args :: Dict) 
    solver.bench_info = TO.TimerOutput("TPFA Solver - $(solver.name)")
    Helpers.reset_verbose()
    Helpers.verbose("== Applying TPFA Scheme over $(solver.name)...", "OUT", solver.verbose)

    step_name :: String = "TPFA Pre-Processing"
    @TO.timeit solver.bench_info step_name begin

    solver.mesh = args["mesh"]
    solver.permeability = args["permeability"]
    if solver.mesh === nothing
        __pre_process!(solver, args["mesh_filename"]) 
    end

    end # Pre-Process
    __verbose(step_name, solver.verbose)

    step_name = "TPFA System Preparation"
    @TO.timeit solver.bench_info step_name begin

    __assemble_faces_transmissibilities!(solver)
    __assemble_TPFA_matrix!(solver, args["source"])

    end # TPFA System Preparation
    __verbose(step_name, solver.verbose)

    step_name = "TPFA Boundary Conditions"
    @TO.timeit solver.bench_info step_name begin

    solver.volumes_with_bc = Vector{Int64}()
    __set_dirichlet_boundary_conditions!(solver, args["dirichlet"])
    __set_neumann_boundary_conditions!(solver, args["neumann"])

    end # TPFA Boundary Conditions
    __verbose(step_name, solver.verbose)

    step_name = "TPFA Solver"
    @TO.timeit solver.bench_info step_name begin

    __solve_TPFA_system!(solver)

    end # TPFA Solver
    __verbose(step_name, solver.verbose)

    step_name = "TPFA Post-Processing"
    @TO.timeit solver.bench_info step_name begin

    # __post_process!(solver, args["analytical"], args["vtk"])

    end # Post-Process
    __verbose(step_name, solver.verbose)
end

function __verbose(step_name :: String, verbose :: Bool)
    
end

function get_error(solver :: Solver) :: Float64
    
end

function __get_normals(solver :: Solver) :: Tuple
    volumes_pairs   = solver.mesh.internal_faces_adj
    internal_faces  = solver.mesh.internal_faces
    faces_nodes     = solver.mesh.faces_connectivity[internal_faces]

    i = solver.mesh.nodes_centers[Helpers.get_column.(faces_nodes, 1)]
    j = solver.mesh.nodes_centers[Helpers.get_column.(faces_nodes, 2)]
    k = solver.mesh.nodes_centers[Helpers.get_column.(faces_nodes, 3)]

    nvols_pairs = size(volumes_pairs)[1]
    nvols = size(volumes_pairs[1])[1]

    volumes_centers = reshape(solver.mesh.volumes_centers[[(volumes_pairs...)...]], 
                             (nvols, nvols_pairs))
    faces_centers   = solver.mesh.faces_centers[internal_faces]

    L = volumes_centers[1, :]
    vL = faces_centers - L

    if nvols > 1
        R = volumes_centers[2, :]
        vR = faces_centers - R
    else
        vR = nothing
    end

    N = LA.cross.(i - j, k - j)
    return N, vL, vR
end

function __assemble_faces_transmissibilities!(solver :: Solver)
    """
    Monta os vetores de transmissibilidade de cada face e os vetores de distancias
    """

    N, vL, vR = __get_normals(solver)
    solver.faces_normals = Helpers.abs_b.(N) ./ LA.norm.(N)

    solver.h_L = abs.(LA.dot.(solver.faces_normals, vL))
    solver.h_R = abs.(LA.dot.(solver.faces_normals, vR))

    KL = solver.permeability[Helpers.get_column.(solver.mesh.internal_faces_adj, 1)]
    KR = solver.permeability[Helpers.get_column.(solver.mesh.internal_faces_adj, 2)]

    # nvols_pairs = size(solver.mesh.internal_faces_adj)[1]
    # KL = reshape(KL, (nvols_pairs, 3, 3))
    # KR = reshape(KR, (nvols_pairs, 3, 3))

    KnL = LA.dot.(solver.faces_normals,  KL .* solver.faces_normals)
    KnR = LA.dot.(solver.faces_normals,  KR .* solver.faces_normals)

    Keq = (KnL .* KnR) ./ ((KnL .* solver.h_R) + (KnR .* solver.h_L))
    solver.faces_trans = Keq .* solver.mesh.faces_areas[solver.mesh.internal_faces]
end

function __assemble_TPFA_matrix!(solver :: Solver, source :: Function)
    row_index_p = Helpers.get_column.(solver.mesh.internal_faces_adj, 1)
    col_index_p = Helpers.get_column.(solver.mesh.internal_faces_adj, 2)
    
    row_index   = [row_index_p..., col_index_p...]
    col_index   = [col_index_p..., row_index_p...]
    data        = [solver.faces_trans..., solver.faces_trans...] .* -1.0

    solver.At_TPFA = SparseArrays.sparse(row_index, col_index, data)

    xv = Helpers.get_column.(solver.mesh.volumes_centers, 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers, 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers, 3)

    solver.bt_TPFA = source(xv, yv, zv, solver.permeability) .* solver.mesh.volumes
    v = vec(sum(solver.At_TPFA, dims = 1))
    solver.At_TPFA = (solver.At_TPFA - LA.Diagonal(solver.At_TPFA)) - LA.Diagonal(v)

    if solver.check
        @assert all(vec(sum(solver.At_TPFA, dims = 1)) .< 1e-15)
    end
    
end

function __set_dirichlet_boundary_conditions!(solver :: Solver, dirichlet :: Function)
    d_volumes = unique(solver.mesh.boundary_faces_adj)
    mask      = indexin(d_volumes, solver.volumes_with_bc) .=== nothing
    d_volumes = d_volumes[mask]
    solver.volumes_with_bc = vcat(solver.volumes_with_bc, setdiff(d_volumes, solver.volumes_with_bc))
    @assert length(d_volumes) > 1 "There is no Dirichlet boundary condition"


    xv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 3)

    d_values = dirichlet(xv, yv, zv, d_volumes, solver.mesh)
    solver.bt_TPFA[d_volumes] = d_values
    solver.At_TPFA[d_volumes, :] .= 0.0
    v = zeros(solver.mesh.nvols)
    v[d_volumes] .= 1.0
    solver.At_TPFA = solver.At_TPFA + LA.Diagonal(v)
    dropzeros!(solver.At_TPFA)

    solver.d_volumes = d_volumes
    solver.d_values  = d_values

end

function __set_neumann_boundary_conditions!(solver :: Solver, neumann :: Function)
    n_volumes = unique(solver.mesh.boundary_faces_adj)
    mask      = indexin(n_volumes, solver.volumes_with_bc) .=== nothing
    n_volumes = n_volumes[mask]
    solver.volumes_with_bc = vcat(solver.volumes_with_bc, setdiff(n_volumes, solver.volumes_with_bc))
    if length(n_volumes) == 0
        return
    end

    xv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 3)

    n_values = neumann(xv, yv, zv, n_volumes, solver.mesh)
    solver.bt_TPFA[n_volumes] += n_values

    solver.n_volumes = n_volumes
    solver.n_values  = n_values
    
end

function __solve_TPFA_system!(solver :: Solver)
    solver.p_TPFA = solver.At_TPFA \ solver.bt_TPFA

    if solver.check
        @assert solver.At_TPFA * solver.p_TPFA ≈ solver.bt_TPFA
    end
end


function __pre_process!(solver :: Solver, mesh_filename :: String)
    mesh = MeshHandler.Mesh(mesh_filename)
    solver.mesh = mesh
end

function post_process!(solver :: Solver, analytical :: Vector{Float64}, vtk :: Bool)
    
end

end # module