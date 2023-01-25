import vtkmodules.vtkCommonCore as vtkCore
import vtkmodules.vtkCommonDataModel as vtkData
import vtkmodules.vtkIOLegacy as vtkIO
import random

# Create a 3x3x3 voxel grid
grid = vtkData.vtkImageData()
grid.SetDimensions(3, 3, 3)

# Create the "test" attribute array
testArray = vtkCore.vtkFloatArray()
testArray.SetName("test")

# Fill the array with random values
for i in range(3*3*3):
    testArray.InsertNextValue(random.random())

# Add the array to the grid's point data
grid.GetPointData().AddArray(testArray)

# Save the grid to a .vtk file
writer = vtkIO.vtkDataSetWriter()
writer.SetFileName("grid.vtk")
writer.SetInputData(grid)
writer.Write()
