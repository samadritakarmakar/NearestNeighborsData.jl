using NearestNeighbors 
include("refData.jl")

abstract type AbstractDataTree end

struct BallDataTree <: AbstractDataTree
    refData::RefData
    tree::BallTree
end

struct KDDataTree <: AbstractDataTree
    refData::RefData
    tree::KDTree
end

function BallDataTree(refData::RefData, dimension::Int64 = 3)
    if refData.points isa AbstractVector
        return BallDataTree(refData, BallTree(reshape(refData.points, dimension, Int64(length(refData.points)/dimension))))
    end
    return BallDataTree(refData, BallTree(refData.points))
end

function BallDataTree(data::Union{Dict, Matrix}, points::Union{Dict, Vector, Matrix})
    refData = RefData(data, points)
    BallDataTree(refData, BallTree(refData.points))
end

function BallDataTree(dataVector::Vector{Float64}, problemDim::Int64, points::Union{Dict, Vector, Matrix})
    refData = RefData(dataVector, problemDim, points)
    BallDataTree(refData, BallTree(refData.points))
end

function KDDataTree(refData::RefData, dimension::Int64 = 3)
    if refData.points isa AbstractVector
        return KDDataTree(refData, BallTree(reshape(refData.points, dimension, Int64(length(refData.points)/dimension))))
    end
    return KDDataTree(refData, BallTree(refData.points))
end

function KDDataTree(data::Union{Dict, Matrix}, points::Union{Dict, Vector, Matrix})
    refData = RefData(data, points)
    KDDataTree(refData, KDTree(refData.points))
end

function KDDataTree(data::Union{Dict, Matrix}, points::Union{Dict, Vector, Matrix})
    refData = RefData(data, points)
    KDDataTree(refData, KDTree(refData.points))
end

function KDDataTree(dataVector::Vector{Float64}, problemDim::Int64, points::Union{Dict, Vector, Matrix})
    refData = RefData(dataVector, problemDim, points)
    KDDataTree(refData, KDTree(refData.points))
end

function knnData(dataTree::AbstractDataTree, point::AbstractVector{Float64}, k::Int64, sortres = true)
    knnIdxs, knnDists = knn(dataTree.tree, point, k, sortres)
    ty1 = typeof(dataTree.refData.dataDict)
    dataVector = Vector{ty1.parameters[2]}(undef, k)
    ty2 = typeof(dataTree.refData.pointToActualIndexMap)
    actualIndexVector = Vector{ty2.parameters[2]}(undef, k)
    for i ∈ 1:k
        dataVector[i] = dataTree.refData.dataDict[knnIdxs[i]]
        actualIndexVector[i] = dataTree.refData.pointToActualIndexMap[knnIdxs[i]]
    end
    return dataVector, knnDists, actualIndexVector
end

function nnData(dataTree::AbstractDataTree, point::AbstractVector{Float64})
    dataVector, knnDists, actualIndexVector = knnData(dataTree, point, 1)
    return dataVector[1], knnDists[1], actualIndexVector[1]
end