using PEPSAD
using ITensors

@testset "test peps" begin
    Nx = 5
    Ny = 5
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Nx, Ny)
    peps = PEPS(sites)

    for ii = 1:Nx-1
        for jj = 1:Ny-1
            inds1 = inds(peps.data[ii, jj])
            inds2 = inds(peps.data[ii, jj+1])
            inds3 = inds(peps.data[ii+1, jj])
            inds4 = inds(peps.data[ii+1, jj+1])
            @assert(length(intersect(inds1, inds2)) == 1)
            @assert(length(intersect(inds1, inds3)) == 1)
            @assert(length(intersect(inds1, inds4)) == 0)
        end
    end
end
