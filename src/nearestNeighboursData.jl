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
    return KDDataTree(refData, KDTree(refData.points))
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

function knnData(dataTree::AbstractDataTree, point::AbstractVector{Float64}, k::Int64, sortres::Bool = true)
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

struct knnUniqueData{T, I}
    dataVector::Vector{T}
    actualIndexVector::Vector{I}
    dataPoints::Matrix{Float64}
end

function knnUniqueData(dataTree::AbstractDataTree, points::AbstractMatrix{Float64}, k::Int64, sortres::Bool = false)
    knnIdxsVectors, knnDists = knn(dataTree.tree, points, k, sortres)
    knnIdxs = unique(sort(vcat(knnIdxsVectors...)))
    total_k = length(knnIdxs)
    ty1 = typeof(dataTree.refData.dataDict)
    dataVector = Vector{ty1.parameters[2]}(undef, total_k)
    ty2 = typeof(dataTree.refData.pointToActualIndexMap)
    actualIndexVector = Vector{ty2.parameters[2]}(undef, total_k)
    pointsMatrix = zeros(size(points, 1), total_k)
    for i ∈ 1:total_k
        dataVector[i] = dataTree.refData.dataDict[knnIdxs[i]]
        actualIndexVector[i] = dataTree.refData.pointToActualIndexMap[knnIdxs[i]]
        pointsMatrix[:,i] .= dataTree.refData.points[:,knnIdxs[i]]
    end
    return knnUniqueData(dataVector, actualIndexVector, pointsMatrix)
end

struct knnMinDistData{T, I}
    dataVector::Vector{T}
    knnDists::Vector{Float64}
    actualIndexVector::Vector{I}
    dataPoints::Matrix{Float64}
end


function knnMinDistData(dataTree::AbstractDataTree, points::AbstractMatrix{Float64}, k::Int64, sortres::Bool = false)
    knnIdxsVectors, knnDists = knn(dataTree.tree, points, k, sortres)
    knnIdxs = unique(sort(vcat(knnIdxsVectors...)))
    total_k = length(knnIdxs)
    ty1 = typeof(dataTree.refData.dataDict)
    dataVector = Vector{ty1.parameters[2]}(undef, total_k)
    ty2 = typeof(dataTree.refData.pointToActualIndexMap)
    actualIndexVector = Vector{ty2.parameters[2]}(undef, total_k)
    pointsMatrix = zeros(size(points, 1), total_k)
    for i ∈ 1:total_k
        dataVector[i] = dataTree.refData.dataDict[knnIdxs[i]]
        actualIndexVector[i] = dataTree.refData.pointToActualIndexMap[knnIdxs[i]]
        pointsMatrix[:,i] .= dataTree.refData.points[:,knnIdxs[i]]
    end
    minDists = zeros(total_k)
    fill!(minDists, Inf64)
    vec = 1
    for knnIdxsVector ∈ knnIdxsVectors
        i = 1
        for knnIndx ∈ knnIdxsVector
            indxRange = searchsorted(knnIdxs, knnIndx)
            if length(indxRange) > 0
                uniqueIndxNo = collect(indxRange)[1]
                minDists[uniqueIndxNo] = min(minDists[uniqueIndxNo], knnDists[vec][i])
            end
            i += 1
        end
        vec += 1
    end
    knnMinDistData(dataVector, minDists, actualIndexVector, pointsMatrix)
end

struct inrangeData{T, I}
    dataVector::Vector{T}
    actualIndexVector::Vector{I}
    dataPoints::Matrix{Float64}
    noOfPointsFound::Int64
end

function inrangeData(dataTree::AbstractDataTree, point::AbstractVector{Float64}, radius::Float64, sortres::Bool = true)
    inrangeIdxs = inrange(dataTree.tree, point, radius, sortres)
    k = length(inrangeIdxs)
    ty1 = typeof(dataTree.refData.dataDict)
    dataVector = Vector{ty1.parameters[2]}(undef, k)
    ty2 = typeof(dataTree.refData.pointToActualIndexMap)
    actualIndexVector = Vector{ty2.parameters[2]}(undef, k)
    pointsMatrix = zeros(length(point), k)
    for i ∈ 1:k
        dataVector[i] = dataTree.refData.dataDict[inrangeIdxs[i]]
        actualIndexVector[i] = dataTree.refData.pointToActualIndexMap[inrangeIdxs[i]]
        pointsMatrix[:,i] .= dataTree.refData.points[:,inrangeIdxs[i]]
    end
    return inrangeData(dataVector, actualIndexVector, pointsMatrix, k)
end
