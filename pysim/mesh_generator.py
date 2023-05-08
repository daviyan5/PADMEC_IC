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

        vols_axis = int(order ** (1/3))
        mn = min(Lx, Ly, Lz) ** 0.4
        mx = max(Lx, Ly, Lz) ** 0.4
        norm = np.sqrt(mn**2 + mx**2)
        mn /= norm
        mx /= norm
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mn / vols_axis)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mx / vols_axis)
        gmsh.option.setNumber('Mesh.MeshSizeMin', mn / vols_axis)
        gmsh.option.setNumber('Mesh.MeshSizeMax', mx / vols_axis)
        gmsh.model.mesh.setTransfiniteAutomatic()
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
    Lx, Ly, Lz = 1, 1, 1
    # Create 20 boxes
    MeshGen.create_box((Lx, Ly, Lz), 1, "teste.msh", False)

if __name__ == "__main__":
    main()