import gmsh
import sys
import os
import numpy as np

# TODO: Better meshing
class MeshGenerator:
    def __init__(self):
        pass
    def create_box(self, box_dimensions : tuple, order : int, filename : str = None, visualize : bool = False) -> str:
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.16)
        if not visualize:
            gmsh.option.setNumber("General.Verbosity", 0)
        else:
            gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("box")

        Lx = box_dimensions[0]
        Ly = box_dimensions[1]
        Lz = box_dimensions[2]

        v1 = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
        gmsh.model.occ.synchronize()

        for c in gmsh.model.getEntities(1):
            if c[1] in [1, 3, 5, 7]:
                gmsh.model.mesh.setTransfiniteCurve(c[1], 2)
            else:
                gmsh.model.mesh.setTransfiniteCurve(c[1], order)
        for s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s[1])
            gmsh.model.mesh.setRecombine(s[0], s[1])
            gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
        gmsh.model.mesh.setTransfiniteVolume(v1)
        gmsh.model.mesh.generate(3)
        
        if filename is None:
            filename = "box.msh"
        filename = os.path.join("mesh", filename)
        gmsh.write(filename)

        if '-nopopup' not in sys.argv and visualize:
            gmsh.fltk.run()

        gmsh.finalize()
        return filename



def main():
    MeshGen = MeshGenerator()
    Lx, Ly, Lz = 6, 4, 1
    # Create 20 boxes
    order = 2
    for i in range(10):
        MeshGen.create_box((Lx, Ly, Lz), order + 1, "box_{}.msh".format(i), False)
        print("Created box_{}.msh".format(i))
        order *= 2

if __name__ == "__main__":
    main()
