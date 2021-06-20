using ITensors

"""
A finite size PEPS type.
"""
mutable struct PEPS
    data::Matrix{ITensor}
    dimx::Int
    dimy::Int
end


PEPS(Nx::Int, Ny::Int) = PEPS(Matrix{ITensor}(undef, Nx, Ny), Nx, Ny)


"""
    PEPS([::Type{ElT} = Float64, sites; linkdims=1)
Construct an PEPS filled with Empty ITensors of type `ElT` from a collection of indices.
Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.
"""
function PEPS(::Type{T}, sites::Matrix{<:Index}; linkdims::Integer = 1) where {T<:Number}
    Ny, Nx = size(sites)
    tensor_grid = Matrix{ITensor}(undef, Ny, Nx)
    # we assume the PEPS at least has size (2,2). Can generalize if necessary
    @assert(Nx >= 2 && Ny >= 2)

    lh = Matrix{Index}(undef, Ny, Nx - 1)
    for ii = 1:(Nx-1)
        for jj = 1:(Ny)
            lh[jj, ii] = Index(linkdims, "Lh,$jj,$ii")
        end
    end
    lv = Matrix{Index}(undef, Ny - 1, Nx)
    for ii = 1:(Nx)
        for jj = 1:(Ny-1)
            lv[jj, ii] = Index(linkdims, "Lv,$jj,$ii")
        end
    end

    # boundary cases
    tensor_grid[1, 1] = emptyITensor(T, lh[1, 1], lv[1, 1], sites[1, 1])
    tensor_grid[1, Nx] = emptyITensor(T, lh[1, Nx-1], lv[1, Nx], sites[1, Nx])
    tensor_grid[Ny, 1] = emptyITensor(T, lh[Ny, 1], lv[Ny-1, 1], sites[Ny, 1])
    tensor_grid[Ny, Nx] = emptyITensor(T, lh[Ny, Nx-1], lv[Ny-1, Nx], sites[Ny, Nx])
    for ii = 2:Nx-1
        tensor_grid[1, ii] =
            emptyITensor(T, lh[1, ii], lh[1, ii-1], lv[1, ii], sites[1, ii])
        tensor_grid[Ny, ii] =
            emptyITensor(T, lh[Ny, ii], lh[Ny, ii-1], lv[Ny-1, ii], sites[Ny, ii])
    end

    # inner sites
    for jj = 2:Ny-1
        tensor_grid[jj, 1] =
            emptyITensor(T, lh[jj, 1], lv[jj, 1], lv[jj-1, 1], sites[jj, 1])
        tensor_grid[jj, Nx] =
            emptyITensor(T, lh[jj, Nx-1], lv[jj, Nx], lv[jj-1, Nx], sites[jj, Nx])
        for ii = 2:Nx-1
            tensor_grid[jj, ii] = emptyITensor(
                T,
                lh[jj, ii],
                lh[jj, ii-1],
                lv[jj, ii],
                lv[jj-1, ii],
                sites[jj, ii],
            )
        end
    end

    return PEPS(tensor_grid, Nx, Ny)
end

PEPS(sites::Matrix{<:Index}, args...; kwargs...) = PEPS(Float64, sites, args...; kwargs...)

function randomizePEPS!(P::PEPS)
    for ii = 1:P.dimx
        for jj = 1:P.dimy
            randn!(P.data[jj, ii])
            normalize!(P.data[jj, ii])
        end
    end
end

# Get the tensor network of <peps|peps>
function inner_network(peps::PEPS)
    network = []
    for tensor in vcat(peps.data...)
        push!(network, tensor)
        indices = inds(tensor)
        bonds = [indices[i] for i = 1:length(indices)-1]
        push!(network, dag(prime(tensor, bonds...)))
    end
    return network
end

# Get the tensor network of <peps|mpo|peps>
# The local MPO specifies the 2-site term of the Hamiltonian
function inner_network(peps::PEPS, mpo::MPO, coordinates::Array)
    @assert(length(mpo) == length(coordinates))
    network = []
    for ii = 1:peps.dimx
        for jj = 1:peps.dimy
            tensor = peps.data[jj, ii]
            push!(network, tensor)
            indices = inds(tensor)
            if (jj => ii) in coordinates
                index = findall(x -> x == (jj => ii), coordinates)
                @assert(length(index) == 1)
                push!(network, mpo.data[index[1]])
                push!(network, dag(prime(tensor, indices...)))
            else
                bonds = [indices[i] for i = 1:length(indices)-1]
                push!(network, dag(prime(tensor, bonds...)))
            end
        end
    end
    return network
end
