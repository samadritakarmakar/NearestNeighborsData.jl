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
        refData.points = reshape(refData.points, dimension, Int64(length(refData.points)/dimension))
        #return BallDataTree(refData, BallTree(refData.points))
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
        refData.points = reshape(refData.points, dimension, Int64(length(refData.points)/dimension))
        #return KDDataTree(refData, BallTree(reshape(refData.points, dimension, Int64(length(refData.points)/dimension))))
    end
    return KDDataTree(refData, BallTree(refData.points))
end

function KDDataTree(data::Union{Dict, Matrix}, points::Union{Dict, Vector, Matrix})
    refData = RefData(data, points)
    KDDataTree(refData, KDTree(refData.points))
end

function KDDataTree(dataVector::Vector{Float64}, problemDim::Int64, points::Union{Dict, Vector, Matrix})
    refData = RefData(dataVector, problemDim, points)
    KDDataTree(refData, KDTree(refData.points))
end

struct knnData{T, I}
    dataVector::Vector{T}
    knnDists::Vector{Float64}
    actualIndexVector::Vector{I}
    dataPoints::Matrix{Float64}
end

function knnData(dataTree::AbstractDataTree, point::AbstractVector{Float64}, k::Int64, sortres = true)
    knnIdxs, knnDists = knn(dataTree.tree, point, k, sortres)
    ty1 = typeof(dataTree.refData.dataDict)
    dataVector = Vector{ty1.parameters[2]}(undef, k)
    ty2 = typeof(dataTree.refData.pointToActualIndexMap)
    actualIndexVector = Vector{ty2.parameters[2]}(undef, k)
    pointsMatrix = zeros(length(point), k)
    for i ∈ 1:k
        dataVector[i] = dataTree.refData.dataDict[knnIdxs[i]]
        actualIndexVector[i] = dataTree.refData.pointToActualIndexMap[knnIdxs[i]]
        pointsMatrix[:,i] .= dataTree.refData.points[:,knnIdxs[i]]
    end
    return knnData(dataVector, knnDists, actualIndexVector, pointsMatrix)
end

struct nnData{T, I}
    data::T
    nnDist::Float64
    actualIndex::I
    dataPoint::Vector{Float64}
end
function nnData(dataTree::AbstractDataTree, point::AbstractVector{Float64})
    knnDataSet = knnData(dataTree, point, 1)
    nnData(knnDataSet.dataVector[1], knnDataSet.knnDists[1], knnDataSet.actualIndexVector[1], knnDataSet.dataPoints[:,1])
end