using PEPSAD
using ITensors
using AutoHOOT

const itensorad = AutoHOOT.ITensorsAD

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
            @test length(intersect(inds1, inds2)) == 1
            @test length(intersect(inds1, inds3)) == 1
            @test length(intersect(inds1, inds4)) == 0
        end
    end
end

@testset "test inner product" begin
    Nx = 3
    Ny = 3
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Ny, Nx)
    peps = PEPS(sites)
    randomizePEPS!(peps)
    inner = inner_network(peps)

    opt_inner = itensorad.generate_optimal_tree(inner)
    out = contract(opt_inner)
    # output is a scalar
    @test size(out) == ()
end

@testset "test inner product with hamiltonian" begin
    Nx = 3
    Ny = 3
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Ny, Nx)
    peps = PEPS(sites)
    randomizePEPS!(peps)

    opsum = OpSum()
    opsum += 0.5, "S+", 1, "S-", 2
    opsum += 0.5, "S-", 1, "S+", 2
    opsum += "Sz", 1, "Sz", 2
    mpo = MPO(opsum, [sites[2, 2], sites[2, 3]])

    inner = inner_network(peps, mpo, [2 => 2, 2 => 3])
    opt_inner = itensorad.generate_optimal_tree(inner)
    out = contract(opt_inner)
    # output is a scalar
    @test size(out) == ()
end
