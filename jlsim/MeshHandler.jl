module MeshHandler


import LinearAlgebra as LA
import TimerOutputs as TO

using PyCall, StaticArrays

function __init__()
        py"""

        import numpy as np
        import os
        import sys

        from preprocessor.meshHandle.finescaleMesh import FineScaleMesh 

        def get_mesh(mesh_filename : str) -> FineScaleMesh:
            
            sys.stdout = open(os.devnull, 'w')
            mesh = FineScaleMesh(mesh_filename, 3)
            sys.stdout.close()
            sys.stdout = sys.__stdout__
    
            return mesh

        def get_volumes_centers(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.volumes.center[:]
        
        def get_faces_centers(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.faces.center[:]
        
        def get_volumes_connectivity(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.volumes.connectivities[:] + 1

        def get_nodes_center(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.nodes.coords[:]

        def get_faces_connectivity(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.faces.connectivities[:] + 1
        
        def get_internal_faces(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.faces.internal[:] + 1
        
        def get_boundary_faces(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.faces.boundary[:] + 1
        
        def get_faces_by_volume(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.volumes.bridge_adjacencies(mesh.volumes.all, 3, 2) + 1

        def get_volumes_adjacent_faces(mesh: FineScaleMesh) -> np.ndarray:
            return mesh.volumes.bridge_adjacencies(mesh.volumes.all, 3, 2) + 1

        def get_internal_faces_adjacent_volumes(mesh: FineScaleMesh) -> np.ndarray:
            faces_index     = mesh.faces.internal[:]
            vols_pairs      = mesh.faces.bridge_adjacencies(faces_index, 2, 3)
            nvols_pairs     = vols_pairs.shape[0]
            faces_centers   = mesh.faces.center[faces_index]
            volumes_centers = mesh.volumes.center[vols_pairs.flatten()]
            volumes_centers = volumes_centers.reshape(nvols_pairs, 2, 3)

            L = volumes_centers[:, 0]

            vL = (L - faces_centers).sum(axis = 1)

            vols_pairs[vL > 0, 0], vols_pairs[vL > 0, 1] = vols_pairs[vL > 0, 1], vols_pairs[vL > 0, 0]
            return vols_pairs + np.array([1, 1])
        
        def get_boundary_faces_adjacent_volumes(mesh: FineScaleMesh) -> np.ndarray:
            faces_index = mesh.faces.boundary[:]
            return mesh.faces.bridge_adjacencies(faces_index, 2, 3) + 1
        
        def write_vtk(mesh : FineScaleMesh, numerical : np.ndarray, analytical : np.ndarray, d_volumes : np.ndarray, filename : str) -> str:
            import os
            meshset = mesh.core.mb.create_meshset()
            mesh.index[:] = np.arange(1, len(numerical) + 1)
            d_v = np.zeros_like(numerical)
            d_v[d_volumes - 1] = 1
            mesh.d_volumes[:] = d_v.astype(np.int)
            mesh.numerical_p[:] = numerical
            mesh.analytical_p[:] = analytical if analytical is not None else np.zeros_like(numerical)
            mesh.core.mb.add_entities(meshset, mesh.core.all_volumes)
            write_filename = os.path.join("vtks", filename)
            mesh.core.mb.write_file(write_filename, [meshset])
            return write_filename
        
        """
end

struct Mesh
        mesh_filename       :: String
        nvols               :: Int64
        nfaces              :: Int64
        
        volumes             :: Vector{T} where T <: Float64
        faces_areas         :: Vector{U} where U <: Float64
        
        volumes_centers     :: Vector{SVector{3, V}} where V <: Float64
        faces_centers       :: Vector{SVector{3, W}} where W <: Float64
        nodes_centers       :: Vector{SVector{3, X}} where X <: Float64
        
        internal_faces      :: Vector{Y} where Y <: Int64
        boundary_faces      :: Vector{Z} where Z <: Int64
        
        volumes_adj         :: Vector{SVector{6, A}} where A <: Int64
        internal_faces_adj  :: Vector{SVector{2, B}} where B <: Int64
        boundary_faces_adj  :: Vector{C} where C <: Int64
        
        vols_connectivity   :: Vector{SVector{8, D}} where D <: Int64
        faces_connectivity  :: Vector{SVector{4, E}} where E <: Int64
    
        
        mesh                :: PyObject
        to                  :: TO.TimerOutput
        
        function Mesh(mesh_filename :: String)
            to = TO.TimerOutput()
            @TO.timeit to "load mesh" mesh = get_mesh(mesh_filename)
            nvols   = length(mesh.volumes.all[:])
            nfaces  = length(mesh.faces.all[:]) 
            
            @TO.timeit to "get volumes centers" begin
            vc_matrix           = py"get_volumes_centers"(mesh)
            volumes_centers_aux = Vector{eltype(vc_matrix)}[eachcol(transpose(vc_matrix))...]
            volumes_centers     = [SVector{3, eltype(vc_matrix)}(vaux) for vaux in volumes_centers_aux]
            end

            @TO.timeit to "get faces centers" begin
            fc_matrix           = py"get_faces_centers"(mesh)
            faces_centers_aux   = Vector{eltype(fc_matrix)}[eachcol(transpose(fc_matrix))...]
            faces_centers       = [SVector{3, eltype(fc_matrix)}(vaux) for vaux in faces_centers_aux]
            end
            
            @TO.timeit to "get nodes centers" begin
            n_matrix            = py"get_nodes_center"(mesh)
            nodes_centers_aux   = Vector{eltype(n_matrix)}[eachcol(transpose(n_matrix))...]
            nodes_centers       = [SVector{3, eltype(n_matrix)}(vaux) for vaux in nodes_centers_aux]
            end

            @TO.timeit to "get faces" begin
            internal_faces      = py"get_internal_faces"(mesh)
            boundary_faces      = py"get_boundary_faces"(mesh)
            end

            @TO.timeit to "get adj faces" begin
            va_matrix           = py"get_volumes_adjacent_faces"(mesh)
            va_aux              = Vector{eltype(va_matrix)}[eachcol(transpose(va_matrix))...]
            volumes_adj         = [SVector{6, eltype(va_matrix)}(vaux) for vaux in va_aux]
            end

            @TO.timeit to "get internal faces adj" begin
            if_matrix           = py"get_internal_faces_adjacent_volumes"(mesh)
            if_aux              = Vector{eltype(if_matrix)}[eachcol(transpose(if_matrix))...]
            internal_faces_adj  = [SVector{2, eltype(if_matrix)}(vaux) for vaux in if_aux]
            end

            @TO.timeit to "get boundary faces adj" begin
            bf_matrix           = py"get_boundary_faces_adjacent_volumes"(mesh)
            bf_aux              = Vector{eltype(bf_matrix)}[eachcol(transpose(bf_matrix))...]
            boundary_faces_adj  = [(bf_aux...)...]
            end

            @TO.timeit to "get vols connectivity" begin
            vco_matrix          = py"get_volumes_connectivity"(mesh)
            vco_aux             = Vector{eltype(vco_matrix)}[eachcol(transpose(vco_matrix))...]
            vols_connectivity   = [SVector{8, eltype(vco_matrix)}(vaux) for vaux in vco_aux]
            end

            @TO.timeit to "get faces connectivity" begin
            fc_matrix           = py"get_faces_connectivity"(mesh)
            fc_aux              = Vector{eltype(fc_matrix)}[eachcol(transpose(fc_matrix))...]
            faces_connectivity  = [SVector{4, eltype(fc_matrix)}(vaux) for vaux in fc_aux]
            end

            @TO.timeit to "get faces areas" begin
            i                   = nodes_centers[fc_matrix[:, 1]]
            j                   = nodes_centers[fc_matrix[:, 2]]
            k                   = nodes_centers[fc_matrix[:, 3]]
            faces_areas         = LA.norm.(LA.cross.(i - j, k - j))
            end

            @TO.timeit to "get volumes" begin
            area_by_adjface     = reshape(faces_areas[reduce(vcat, volumes_adj)], 6, nvols)
            volumes             = prod.(eachcol(area_by_adjface)) .^ (1/4)
            end
            TO.disable_timer!(to)
            new(mesh_filename, nvols, nfaces,
                volumes, faces_areas,
                volumes_centers, faces_centers, nodes_centers,
                internal_faces, boundary_faces,
                volumes_adj, internal_faces_adj, boundary_faces_adj,
                vols_connectivity, faces_connectivity,
                mesh, to)

        end
end

function Base.show(io :: IO, m :: Mesh)
    println(io, "$(m.nvols)-element and $(m.nfaces)-faces from '$(m.mesh_filename)'")
    println(io, "Mesh - Time to build: \t\t$(round(TO.tottime(m.to) / 1e9, digits = 3)) seconds")
    print(io, "Mesh - Allocated memory: \t$(round(TO.totallocated(m.to) / 1e6, digits = 3)) MBytes")
end

function get_mesh(mesh_filename :: String) :: PyObject
    return py"get_mesh"(mesh_filename)
end

function write_vtk(mesh :: Mesh, numerical :: Vector{Float64}, analytical :: Union{Nothing, Vector{Float64}}, d_volumes :: Vector{Int64}, filename :: String)
    vtk_filename = py"write_vtk"(mesh.mesh, numerical, analytical, d_volumes, filename)
    return vtk_filename
end


end # module


#import .MeshHandler

#a = MeshHandler.Mesh("./mesh/box_9acc.msh")

