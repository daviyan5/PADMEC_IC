
module MeshGenerator

import Gmsh: gmsh
import Base
function create_box(box_dimensions :: Tuple, order :: Int64, filename :: String = nothing, visualize :: Bool = false) :: String
    gmsh.initialize()
    if !visualize
        gmsh.option.setNumber("General.Verbosity", 0)
    end

    gmsh.model.add("box")

    Lx = box_dimensions[1]
    Ly = box_dimensions[2]
    Lz = box_dimensions[3]

    # Create the box
    v1 = gmsh.model.occ.add_box(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    vols_axis = Int16(order ^ (1/3))
    mn = min(Lx, Ly, Lz) ^ (0.4)
    mx = max(Lx, Ly, Lz) ^ (0.4)
    norm = sqrt(mn ^ 2 + mx ^ 2)
    mn = mn / norm
    mx = mx / norm

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mn / vols_axis)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mx / vols_axis)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mn / vols_axis)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mx / vols_axis)
    gmsh.model.mesh.setTransfiniteAutomatic()
    gmsh.model.mesh.generate(3)

    if filename === nothing                     # === Verifica se os objetos são iguais
        filename = "box.msh"
    end
    

    filename = Base.joinpath("mesh", filename)
    gmsh.write(filename)
    gmsh.finalize()
    return filename

end

end # module MeshGenerator

import .MeshGenerator: create_box
function testing()
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    create_box((Lx, Ly, Lz), 1, "box.msh", true)
    
end

#testing()