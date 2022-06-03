 using RapidFEM
 include("../nearestNeighboursData.jl")

 function trial()
     mesh::Mesh = readMesh("annulus2D_O1.msh")
     problemDim = 3
     u = rand(problemDim*mesh.noOfNodes)
     @time RefData(u, problemDim, mesh.Nodes)
 end
