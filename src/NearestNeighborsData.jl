module NearestNeighborsData
using NearestNeighbors

include("nearestNeighboursData.jl")

#from refData.jl
export RefData, addToRefData!
#from nearestNeighboursData
export AbstractDataTree, BallDataTree, KDDataTree, knnData, nnData, inrangeData
export knnUniqueData, knnMinDistData

end # module
