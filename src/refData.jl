mutable struct RefData{D, I}
    dataDict::Dict{Int64, D}
    #points::Union{Vector{AbstractVector{Float64}}, Vector{Vector{Float64}}}
    #AbstractVector{Float64} is there to allow appending new vectors
    points::Union{AbstractMatrix{Float64}, AbstractVector{Float64}}
    pointToActualIndexMap::Dict{Int64, I}
end

function RefData{D, I}() where D where I
    RefData(Dict{Int64, D}(), Vector{Float64}(), Dict{Int64, I}())
end

function RefData{D}() where D
    RefData(Dict{Int64, D}(), Vector{Float64}(), Dict{Int64, Int64}())
end

function addToRefData!(refData::RefData{D}, data::D, point::AbstractVector{Float64}) where {D}
    append!(refData.points, point)
    matrixSize2 = Int64(length(refData.points)/length(point))
    refData.dataDict[matrixSize2] = data
    refData.pointToActualIndexMap[matrixSize2] = length(point)
end

function addToRefData!(refData::RefData{D}, data::D, point::AbstractVector{Float64}, index::I) where {D, I}
    append!(refData.points, point)
    matrixSize2 = Int64(length(refData.points)/length(point))
    refData.dataDict[matrixSize2] = data
    refData.pointToActualIndexMap[matrixSize2] = index
end

function addToRefData!(refData::RefData{D}, data::D, index::Int64) where {D}
    refData.dataDict[index] = data
    refData.pointToActualIndexMap[index] = index
end

function RefData(dataVectorOrMap::Union{Dict{Int64, D}, Vector{D}}, 
    points::Union{Vector{Vector{Float64}}, Matrix{Float64}}) where {D}

    refData = RefData{D}() 
    if points isa Matrix
        indices = 1:size(points, 2)
        refData.points = points
    else
        indices = 1:length(indices)
        refData.points = hcat(points...)
    end
    for index ∈ indices
        refData.dataDict[index] = dataVectorOrMap[index]
        refData.pointToActualIndexMap[index] = index
    end
    return refData
end

function RefData(dataMatrix::Matrix{Float64}, 
    points::Union{Vector{AbstractVector{Float64}}, Vector{Vector{Float64}}, Matrix{Float64}})

    refData = RefData{Vector{Float64}}()
    dataRefVector = Vector{AbstractVector{Float64}}(undef, size(dataMatrix, 2))
    for node ∈ 1:length(dataRefVector)
        refData.dataDict[node] = @view dataMatrix[:, node]
        refData.pointToActualIndexMap[node] = node
    end
    if( points isa Vector{Vector{Float64}} || points isa Vector{AbstractVector{Float64}})
        refData.points = hcat(points...)
    else
        refData.points = points
    end
    return refData
end


function RefData(dataVectorOrMap::Union{Dict{I, D}, Vector{D}}, 
    indexToVectorMap::Union{Dict{I, AbstractVector{Float64}}, Dict{I, Vector{Float64}}}) where {D, I}

    if dataVectorOrMap isa Dict
        indices = keys(dataVectorOrMap)
    else
        indices = 1:length(dataVectorOrMap)
    end
    refData = RefData{D, I}()
    lastIndexSize = 0
    for index ∈ indices
        addToRefData!(refData, dataVectorOrMap[index], indexToVectorMap[index], index)
        lastIndexSize = length(indexToVectorMap[index])
    end
    refData.points = reshape(refData.points, lastIndexSize, Int64(length(refData.points)/lastIndexSize))
    return refData
end

function RefData(dataVector::Vector{Float64}, problemDim::Int64,
    indexToVectorMap::Union{Dict{I, AbstractVector{Float64}}, Dict{I, Vector{Float64}}})  where I

    dataRefVector = Vector{AbstractVector{Float64}}(undef, Int64(length(dataVector)/problemDim))
    for node ∈ 1:length(dataRefVector)
        dataRefVector[node] = @view dataVector[problemDim*(node-1)+1:problemDim*node]
    end
    return RefData(dataRefVector, indexToVectorMap)
end


