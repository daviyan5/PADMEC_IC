import gmsh
import sys

def legacy_create_box(Lx, Ly, Lz, lc, filename=None, visualize=False):
    gmsh.initialize()
    if not visualize:
        gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("box")

    # Create points
    for i in range(2):
        for j in range(2):
            for k in range(2):
                index = i * 4 + j * 2 + k
                gmsh.model.geo.addPoint(i * Lx, j * Ly, k * Lz, lc, index)
    
    # Create lines
    gmsh.model.geo.addLine(0, 1, 0)
    gmsh.model.geo.addLine(0, 2, 1)
    gmsh.model.geo.addLine(0, 4, 2)
    gmsh.model.geo.addLine(1, 3, 3)
    gmsh.model.geo.addLine(1, 5, 4)
    gmsh.model.geo.addLine(2, 3, 5)
    gmsh.model.geo.addLine(2, 6, 6)
    gmsh.model.geo.addLine(3, 7, 7)
    gmsh.model.geo.addLine(4, 5, 8)
    gmsh.model.geo.addLine(4, 6, 9)
    gmsh.model.geo.addLine(5, 7, 10)
    gmsh.model.geo.addLine(6, 7, 11)

    # Create surfaces
    c1 = gmsh.model.geo.addCurveLoop([0,3,-5,-1])
    s1 = gmsh.model.geo.addPlaneSurface([c1])

    c2 = gmsh.model.geo.addCurveLoop([1,6,-9,-2])
    s2 = gmsh.model.geo.addPlaneSurface([c2])

    c3 = gmsh.model.geo.addCurveLoop([0, 4, -8, -2])
    s3 = gmsh.model.geo.addPlaneSurface([c3])

    c4 = gmsh.model.geo.addCurveLoop([8,10,-11,-9])
    s4 = gmsh.model.geo.addPlaneSurface([c4])

    c5 = gmsh.model.geo.addCurveLoop([5,7,-11,-6])
    s5 = gmsh.model.geo.addPlaneSurface([c5])

    c6 = gmsh.model.geo.addCurveLoop([3,7,-10,-4])
    s6 = gmsh.model.geo.addPlaneSurface([c6])

    l1 = gmsh.model.geo.addSurfaceLoop([s1,s2,s3,s4,s5,s6])
    v1 = gmsh.model.geo.addVolume([l1])

    #gmsh.model.geo.mesh.setTransfiniteCurve(c1, 10)
    #gmsh.model.geo.mesh.setTransfiniteCurve(c2, 10)
    #gmsh.model.geo.mesh.setTransfiniteCurve(c3, 10)
    #gmsh.model.geo.mesh.setTransfiniteCurve(c4, 10)
    #gmsh.model.geo.mesh.setTransfiniteCurve(c5, 10)
    #gmsh.model.geo.mesh.setTransfiniteCurve(c6, 10)
#
    #gmsh.model.geo.mesh.setTransfiniteSurface(s1)
    #gmsh.model.geo.mesh.setTransfiniteSurface(s2)
    #gmsh.model.geo.mesh.setTransfiniteSurface(s3)
    #gmsh.model.geo.mesh.setTransfiniteSurface(s4)
    #gmsh.model.geo.mesh.setTransfiniteSurface(s5)
    #gmsh.model.geo.mesh.setTransfiniteSurface(s6)
    #
    #gmsh.model.geo.mesh.setTransfiniteVolume(v1)

    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim = 3)

    if filename is None:
        filename = "box.msh"
    #gmsh.option.setNumber("Mesh.RecombineAll", 1)
    #gmsh.option.setNumber("Mesh.RecombineAll", 2)
    #gmsh.option.setNumber("Mesh.RecombineAll", 3)
    #gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', 0.001)
    #gmsh.model.mesh.setTransfiniteAutomatic()

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.16)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv and visualize:
        gmsh.fltk.run()
    
    gmsh.finalize()
    return filename

def create_box(Lx, Ly, Lz, nvols_order, visualize = False, filename = None):
    gmsh.initialize()

    if not visualize:
        gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("box")

    
    v1 = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz, tag = 1)
    gmsh.model.occ.synchronize()

    
    #nvols = int(nvols_order ** (1/3)) + 2
    #for c in gmsh.model.getEntities(1):
    #    gmsh.model.mesh.setTransfiniteCurve(c[1], nvols)
    #for s in gmsh.model.getEntities(2):
    #    gmsh.model.mesh.setTransfiniteSurface(s[1])
    #    gmsh.model.mesh.setRecombine(s[0], s[1])
    #    gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
    #gmsh.model.mesh.setTransfiniteVolume(v1)

    mn = ((Lx * Ly * Lz) / nvols_order) ** (1/3)
    mx = ((Lx * Ly * Lz) / nvols_order) ** (1/3)
    gmsh.option.setNumber('Mesh.MeshSizeMin', mn)
    gmsh.option.setNumber('Mesh.MeshSizeMax', mx)

    gmsh.model.mesh.setTransfiniteAutomatic()
    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.16)

    if filename is None:
        filename = "box.msh"

    gmsh.write(filename)

    if '-nopopup' not in sys.argv and visualize:
        gmsh.fltk.run()

    gmsh.finalize()
    return filename


if __name__ == "__main__": 
    Lx, Ly, Lz = 6, 10, 0.01
    nvols_order = 10000
    #legacy_create_box(Lx, Ly, Lz, 0.1, visualize=True)
    create_box(Lx, Ly, Lz, nvols_order, visualize=True)