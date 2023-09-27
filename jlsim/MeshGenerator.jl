
module MeshGenerator

import Gmsh: gmsh
import Base
function create_box(box_dimensions :: Tuple, order :: Int64, filename :: String = nothing, visualize :: Bool = false) :: Tuple
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.16)
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

    vols_axis = round(order ^ (1/(2.7)))
    mn = min(Lx, Ly, Lz) ^ (0.5)
    mx = max(Lx, Ly, Lz) ^ (0.5)
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
    i = 0
    n = 0
    for s in eachline(filename)
        if i == 4
            n = parse(Int64, s)
            break
        end
        i += 1
    end
    gmsh.finalize()
    return filename, n

end

end # module MeshGenerator

import .MeshGenerator: create_box
function testing()
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    prev_n = 0
    inc = 1.3
    prev_order = 1 / (inc - 1)
    for i in 0:20
        order = Int64(round(prev_order * inc))
        n = 0
        while true
            println("[$(i)] \t Generating box $(i) with order $(order)")
            T = create_box((Lx, Ly, Lz), order, "box_$(i)acc.msh", false)
            n = T[2]
            println("[$(i)] \t Number of nodes: $(n) / $(prev_n)")
            if n == prev_n
                println("[$(i)] \t Same number of nodes, retrying")
                order = order * inc
                order = Int64(round(order))
            else
                println("[$(i)] \t Different number of nodes")
                break
            end
            println()
            
        end
        prev_order = order
        prev_n = n
    end
    
end 

#testing()