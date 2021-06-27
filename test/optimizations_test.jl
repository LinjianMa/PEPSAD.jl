using PEPSAD, ITensors, AutoHOOT, Zygote

const itensorad = AutoHOOT.ITensorsAD

@testset "test monotonic loss decrease of gradient descent" begin
    Nx = 2
    Ny = 3
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Ny, Nx)
    peps = PEPS(sites; linkdims = 10)
    randomizePEPS!(peps)
    H_local = localham(Model("tfim"), sites; h = 1.0)
    losses = gradient_descent(peps, H_local, stepsize = 0.005, num_sweeps = 50)
    for i = 1:length(losses)-1
        @test losses[i] >= losses[i+1]
    end
end

@testset "test inner product gradient" begin
    Nx = 2
    Ny = 2
    sites = siteinds("S=1/2", Nx * Ny)
    sites = reshape(sites, Ny, Nx)
    peps = PEPS(sites; linkdims = 2)
    randomizePEPS!(peps)
    function loss(peps::PEPS)
        peps_prime = PEPSAD.prime(peps; ham = false)
        peps_prime_ham = PEPSAD.prime(peps; ham = true)
        network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, [])
        variables = extract_data([peps, peps_prime])
        inners = itensorad.batch_tensor_contraction(network_list, variables...)
        return itensorad.scalar(sum(inners))
    end
    g = gradient(loss, peps)
    inner = inner_network(peps, PEPSAD.prime(peps; ham = false))
    g_true_first_site = contract(inner[2:length(inner)])
    g_true_first_site = 2 * g_true_first_site
    @test isapprox(norm(g[1].data[1, 1]), norm(g_true_first_site))
end
