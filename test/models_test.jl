using PEPSAD
using ITensors

@testset "test local hamiltonian builder" begin
    Nx = 2
    Ny = 3
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Ny, Nx)
    H = mpo(Model("tfim"), sites; h = 1.0)
    H_local = localham(Model("tfim"), sites; h = 1.0)
    @test checklocalham(H_local, H, sites)
end
