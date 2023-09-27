module TPFASolver
import LinearAlgebra as LA
using MKL
using Pardiso

# include("Helpers.jl")     # Only if called directly
# include("MeshHandler.jl") # Only if called directly

import ..MeshHandler
import ..Helpers

import Pardiso as PS

import TimerOutputs as TO
import LinearSolve as LS

using PyCall, StaticArrays, SparseArrays


const StringNull        = Union{String, Nothing}
const BoolNull          = Union{Bool, Nothing}
const DictNull          = Union{Dict, Nothing}
const TupleNull         = Union{Tuple, Nothing}
const IntNull           = Union{Int64, Nothing}
const FloatNull         = Union{Float64, Nothing}

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
    mesh            :: MeshNull                                         # Mesh object

    permeability    :: Union{Vector{SMatrix{3, 3, Float64}}, Nothing}   # Permeability tensor flattened
    faces_normals   :: Union{Vector{SVector{3, Float64}}, Nothing}      # Coordinates of normal vectors internal faces-wise
    faces_trans     :: FloatVectorNull                                  # Transmissibility of each face

    h_L             :: FloatVectorNull                                  # Distance from face_centers to left volume
    h_R             :: FloatVectorNull                                  # Distance from face_centers to right volume

    row_index       :: IntVectorNull                                    # Row index of the TPFA matrix
    col_index       :: IntVectorNull                                    # Column index of the TPFA matrix
    data            :: FloatVectorNull                                  # Data of the TPFA matrix

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
    n_volumes       :: IntVectorNull                                    # Volumes with Neumann boundary conditions

    bench_info      :: TimerOutputNull                                  # Object with times and memories
    vtk_filename    :: StringNull                                       # Name of the vtk file to be written
    error           :: FloatNull                                        # Error of the solution

    times           :: DictNull
    memory          :: FloatNull

    function Solver(verbose::Bool = true, check::Bool = false, name::String = "MESH")
        new(verbose, check, name, nothing,          # verbose, check, name, mesh,
            nothing, nothing, nothing,              # permeability, faces_normals, faces_trans
            nothing, nothing,                       # h_L, h_R
            nothing, nothing, nothing,              # row_index, col_index, data
            nothing, nothing, nothing,              # At_TPFA, bt_TPFA, p_TPFA
            nothing, nothing, nothing,              # irel, numerical_p, analytical_p
            nothing, nothing,                       # faces_with_bc, volumes_with_bc
            nothing, nothing, nothing,              # d_faces, d_values, d_volumes
            nothing, nothing, nothing,              # n_faces, n_values, n_volumes
            nothing, nothing, nothing,              # bench_info, vtk_filename, error
            nothing, nothing)                       # times, memory
    end

end

function Base.show(io :: IO, m :: Solver)
    println(io, "Solver: '$(m.name)' - Verbose Mode: $(m.verbose) - Check Solution: $(m.check)")
    println(io, m.p_TPFA === nothing ? "No solution available." : "Solution available.")
    println("Mesh:")
    show(io, m.mesh)
    println()
    println(io, "TPFA - Time to build: \t\t$(round(TO.tottime(m.bench_info) / 1e9, digits = 3)) seconds")
    println(io, "TPFA - Allocated memory: \t$(round(TO.totallocated(m.bench_info) / 1e6, digits = 3)) MBytes")
    print(io, "Error: \t $(m.error)")
end
function get_times_keys() :: Vector{String}
    return ["Pre-Processing", "TPFA System Preparation", "TPFA Boundary Conditions", "TPFA System Solving"]
end
function solve_TPFA!(solver :: Solver, args :: Dict)
    solver.bench_info = TO.TimerOutput("TPFA System Solving - $(solver.name)")
    # Helpers.reset_verbose()
    Helpers.verbose("== Applying TPFA Scheme over $(solver.name)...", "OUT", solver.verbose)

    step_name :: String = "Pre-Processing"
    @TO.timeit solver.bench_info step_name begin

    solver.mesh = args["mesh"]
    solver.permeability = args["permeability"]
    if solver.mesh === nothing
        __pre_process!(solver, args["mesh_filename"]) 
    end

    end # Pre-Process
    __verbose(solver, step_name, solver.verbose)

    step_name = "TPFA System Preparation"
    @TO.timeit solver.bench_info step_name begin
    @TO.timeit solver.bench_info "TPFA System Preparation - Assemble Transmissibilities" begin
    __assemble_faces_transmissibilities!(solver)
    end # Assemble Transmissibilities
    @TO.timeit solver.bench_info "TPFA System Preparation - Assemble TPFA Matrix" begin
    __assemble_TPFA_matrix!(solver, args["source"])
    end # Assemble TPFA Matrix
    end # TPFA System Preparation
    __verbose(solver, step_name, solver.verbose)

    step_name = "TPFA Boundary Conditions"
    @TO.timeit solver.bench_info step_name begin

    solver.volumes_with_bc = Vector{Int64}()
    __set_dirichlet_boundary_conditions!(solver, args["dirichlet"])
    __set_neumann_boundary_conditions!(solver, args["neumann"])

    end # TPFA Boundary Conditions
    __verbose(solver, step_name, solver.verbose)

    step_name = "TPFA System Solving"
    @TO.timeit solver.bench_info step_name begin

    __solve_TPFA_system!(solver)
    if solver.check == true
        Helpers.verbose("TPFA Solution is consistent!", "CHK", solver.verbose)
    end

    end # TPFA System Solving
    __verbose(solver, step_name, solver.verbose)

    step_name = "Post-Processing"
    @TO.timeit solver.bench_info step_name begin

    solver.analytical_p = haskey(args, "analytical") ? args["analytical"] : nothing
    __get_error!(solver)
    __post_process!(solver, haskey(args, "vtk") ? args["vtk"] : false)

    solver.times = Dict()
    for key in get_times_keys()
        solver.times[key] = 0.0
        solver.times[key] += TO.time(solver.bench_info[key]) / 1e9
    end
    solver.times["Total Time"] = TO.tottime(solver.bench_info) / 1e9
    solver.memory = TO.totallocated(solver.bench_info) / 1e6
    end # Post-Process
    __verbose(solver, step_name, solver.verbose)
    #TO.complement!(solver.bench_info)
end

function __verbose(solver :: Solver, step_name :: String, verbose :: Bool)
    """
    Steps = "Pre-Processing","TPFA System Preparation", 
            "TPFA Boundary Conditions", "TPFA System Solving", "Post-Processing" 
    """
    if verbose == false
        return
    end
    function avg(x :: Vector)
        return sum(x) / length(x)
    end 
    time = TO.time(solver.bench_info[step_name]) / 1e9
    time = round(time, digits = 5)
    memory = TO.allocated(solver.bench_info[step_name]) / 1e6
    Helpers.verbose("== Done with $(step_name)", "OUT")
    Helpers.verbose("Time for step '$(step_name)': $(time) s", "INFO")
    Helpers.verbose("Allocated memory for step '$(step_name)': $(memory) MBytes", "INFO")
    if step_name == "Pre-Processing"
        Helpers.verbose("Nº of volumes: $(solver.mesh.nvols)", "INFO")
        Helpers.verbose("Average volume: $(avg(solver.mesh.volumes))", "INFO")
        Helpers.verbose("Nº of faces: $(solver.mesh.nfaces)", "INFO")
        Helpers.verbose("Average area: $(avg(solver.mesh.faces_areas))", "INFO")

    elseif step_name == "TPFA System Preparation"
        mx, argmx = findmax(solver.faces_trans)
        mn, argmn = findmin(solver.faces_trans)
        mx, mn = round(mx, digits = 5), round(mn, digits = 5)
        mean = round(avg(solver.faces_trans), digits = 5)
        Helpers.verbose("Avg transmissibility: $(mean)", "INFO")
        Helpers.verbose("Max transmissibility: $(mx) - Face $(argmx)", "INFO")
        Helpers.verbose("Min transmissibility: $(mn) - Face $(argmn)", "INFO")
        mx, argmx = findmax(solver.bt_TPFA)
        mn, argmn = findmin(solver.bt_TPFA)
        mx, mn = round(mx, digits = 5), round(mn, digits = 5)
        mean = round(avg(solver.bt_TPFA), digits = 5)
        Helpers.verbose("Avg Source: $(mean)", "INFO")
        Helpers.verbose("Max Source: $(mx) - Volume $(argmx)", "INFO")
        Helpers.verbose("Min Source: $(mn) - Volume $(argmn)", "INFO")

    elseif step_name == "TPFA Boundary Conditions"
        Helpers.verbose("Nº of volumes with Dirichlet BC: $(length(solver.d_volumes))", "INFO")
        mx, argmx = findmax(solver.d_values)
        mn, argmn = findmin(solver.d_values)
        mx, mn = round(mx, digits = 5), round(mn, digits = 5)
        mean = round(avg(solver.d_values), digits = 5)
        Helpers.verbose("Avg Dirichlet BC: $(mean)", "INFO")
        Helpers.verbose("Max Dirichlet BC: $(mx) - Volume $(argmx)", "INFO")
        Helpers.verbose("Min Dirichlet BC: $(mn) - Volume $(argmn)", "INFO")
        try
            Helpers.verbose("Nº of volumes with Neumann BC: $(length(solver.n_volumes))", "INFO")
            mx, argmx = findmax(solver.n_values)
            mn, argmn = findmin(solver.n_values)
            mx, mn = round(mx, digits = 5), round(mn, digits = 5)
            mean = round(avg(solver.n_values), digits = 5)
            Helpers.verbose("Avg Neumann BC: $(mean)", "INFO")
            Helpers.verbose("Max Neumann BC: $(mx) - Volume $(argmx)", "INFO")
            Helpers.verbose("Min Neumann BC: $(mn) - Volume $(argmn)", "INFO")
        catch
            Helpers.verbose("No Neumann BC", "INFO")
        end

    elseif step_name == "TPFA System Solving"
        mx, argmx = findmax(solver.p_TPFA)
        mn, argmn = findmin(solver.p_TPFA)
        mx, mn = round(mx, digits = 5), round(mn, digits = 5)
        mean = round(avg(solver.p_TPFA), digits = 5)
        Helpers.verbose("Avg Pressure: $(mean)", "INFO")
        Helpers.verbose("Max Pressure: $(mx) - Volume $(argmx)", "INFO")
        Helpers.verbose("Min Pressure: $(mn) - Volume $(argmn)", "INFO")
        Helpers.verbose("Machine Epsilon: $(eps())", "INFO")

        
    elseif step_name == "Post-Processing"
        Helpers.verbose("VTK Filename = $(solver.vtk_filename)", "INFO")
        if solver.analytical_p !== nothing
            mx, argmx = findmax(solver.analytical_p)
            mn, argmn = findmin(solver.analytical_p)
            mx, mn = round(mx, digits = 5), round(mn, digits = 5)
            mean = round(avg(solver.analytical_p), digits = 5)
            Helpers.verbose("Avg Analytical Pressure: $(mean)", "INFO")
            Helpers.verbose("Max Analytical Pressure: $(mx) - Volume $(argmx)", "INFO")
            Helpers.verbose("Min Analytical Pressure: $(mn) - Volume $(argmn)", "INFO")

            mx, argmx = findmax(solver.irel)
            mn, argmn = findmin(solver.irel)
            mx, mn = round(mx, digits = 5), round(mn, digits = 5)
            mean = round(avg(solver.irel), digits = 5)
            Helpers.verbose("Avg I²rel: $(mean)", "INFO")
            Helpers.verbose("Max I²rel: $(mx) - Volume $(argmx)", "INFO")
            Helpers.verbose("Min I²rel: $(mn) - Volume $(argmn)", "INFO")
        end
    end
end

function __get_error!(solver :: Solver)
    if solver.analytical_p === nothing
        return
    end
    solver.irel = sqrt.((solver.analytical_p - solver.p_TPFA).^2 .* solver.mesh.volumes ./ 
                        (solver.analytical_p.^2 .* solver.mesh.volumes))
    solver.error = sqrt(sum((solver.analytical_p - solver.p_TPFA).^2 .* solver.mesh.volumes) ./
                        sum(solver.analytical_p.^2 .* solver.mesh.volumes))
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

    volumes_centers_flat :: Vector{SVector} = solver.mesh.volumes_centers[[(volumes_pairs...)...]]
    volumes_centers :: Matrix{SVector} = reshape(volumes_centers_flat, 
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
    
    diags = [1:solver.mesh.nvols]
    solver.row_index   = [row_index_p..., col_index_p..., (diags...)...]
    solver.col_index   = [col_index_p..., row_index_p..., (diags...)...]
    solver.data        = [solver.faces_trans..., solver.faces_trans..., (zeros(solver.mesh.nvols))...] .* -1.0
    lines_sum          = zeros(solver.mesh.nvols)
    
    lv = @view lines_sum[solver.row_index]
    lv .+= solver.data
    diag_index = findall(solver.row_index .== solver.col_index)
    diag_vals  = solver.row_index[diag_index]
    solver.data[diag_index] .= -lines_sum[diag_vals]

    xv = Helpers.get_column.(solver.mesh.volumes_centers, 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers, 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers, 3)

    solver.bt_TPFA = source(xv, yv, zv, solver.permeability) .* solver.mesh.volumes
    
    if solver.check
        new_lines_sum = zeros(solver.mesh.nvols)
        new_lv = @view new_lines_sum[solver.row_index]
        new_lv .+= solver.data[solver.row_index]
        @assert all(new_lv .< 1e-10)
    end
    
end

function __set_dirichlet_boundary_conditions!(solver :: Solver, dirichlet :: Function)
    d_volumes = unique(solver.mesh.boundary_faces_adj)
    mask      = indexin(d_volumes, solver.volumes_with_bc) .=== nothing
    d_volumes = d_volumes[mask]
    xv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers[d_volumes], 3)

    d_values = dirichlet(xv, yv, zv, d_volumes, solver.mesh)
    mask     = findall(d_values .!== nothing)
    d_volumes = d_volumes[mask]
    d_values  = d_values[mask]
    solver.volumes_with_bc = vcat(solver.volumes_with_bc, setdiff(d_volumes, solver.volumes_with_bc))
    
    @assert length(d_volumes) > 1 "There is no Dirichlet boundary condition"

    solver.bt_TPFA[d_volumes] = d_values

    bool_idx = indexin(solver.row_index, d_volumes) .!== nothing
    d_index = findall(bool_idx)
    solver.data[d_index] .= 0.0
    diag_index = findall((solver.row_index .== solver.col_index) .&& (bool_idx))
    solver.data[diag_index] .= 1.0
    solver.d_volumes = d_volumes
    solver.d_values  = d_values

end

function __set_neumann_boundary_conditions!(solver :: Solver, neumann :: Function)
    n_volumes = unique(solver.mesh.boundary_faces_adj)
    mask      = indexin(n_volumes, solver.volumes_with_bc) .=== nothing
    n_volumes = n_volumes[mask]
    xv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 1)
    yv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 2)
    zv = Helpers.get_column.(solver.mesh.volumes_centers[n_volumes], 3)

    n_values = neumann(xv, yv, zv, n_volumes, solver.mesh)
    mask     = findall(n_values .!== nothing)
    n_volumes = n_volumes[mask]
    n_values  = n_values[mask]
    solver.volumes_with_bc = vcat(solver.volumes_with_bc, setdiff(n_volumes, solver.volumes_with_bc))
    if length(n_volumes) == 0
        return
    end

    
    solver.bt_TPFA[n_volumes] += n_values

    solver.n_volumes = n_volumes
    solver.n_values  = n_values
    
end

function __solve_TPFA_system!(solver :: Solver)
    
    solver.At_TPFA = SparseArrays.sparse(solver.row_index, solver.col_index, solver.data, 
                                         solver.mesh.nvols, solver.mesh.nvols)
    
    solver.p_TPFA = zeros(length(solver.bt_TPFA))
    #PS.solve!(ps, solver.p_TPFA, solver.At_TPFA, solver.bt_TPFA)
    prob = LS.LinearProblem(solver.At_TPFA, solver.bt_TPFA)
    #solver.p_TPFA = solver.At_TPFA \ solver.bt_TPFA
    solver.p_TPFA = LS.solve(prob, LS.MKLPardisoFactorize())
    if solver.check
        @assert solver.At_TPFA * solver.p_TPFA ≈ solver.bt_TPFA
    end
end


function __pre_process!(solver :: Solver, mesh_filename :: String)
    mesh = MeshHandler.Mesh(mesh_filename)
    solver.mesh = mesh
end

function __post_process!(solver :: Solver, vtk :: Bool)
    if vtk == false
        return
    end
    solver.vtk_filename = "mesh_$(solver.name)_$(solver.mesh.nvols)_TPFA.vtk"
    solver.vtk_filename = MeshHandler.write_vtk(solver.mesh, solver.p_TPFA, solver.analytical_p, solver.d_volumes, solver.vtk_filename)
end

end # module