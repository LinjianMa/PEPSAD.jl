using ITensors, AutoHOOT, Zygote, OptimKit
using Zygote: @adjoint

const itensorad = AutoHOOT.ITensorsAD
const ad = AutoHOOT.autodiff
const gops = AutoHOOT.graphops

"""Generate an array of networks representing inner products, <p|H_1|p>, ..., <p|H_n|p>, <p|p>
Parameters
----------
peps: a peps network with datatype PEPS
peps_prime: prime of peps used for inner products
peps_prime_ham: prime of peps used for calculating expectation values
Hlocal: An array of MPO operators with datatype LocalMPO
Returns
-------
An array of networks.
"""
function generate_inner_network(
    peps::PEPS,
    peps_prime::PEPS,
    peps_prime_ham::PEPS,
    Hlocal::Array,
)
    network_list = []
    for H_term in Hlocal
        inner = inner_network(
            peps,
            peps_prime,
            peps_prime_ham,
            H_term.mpo,
            [H_term.coord1, H_term.coord2],
        )
        network_list = vcat(network_list, [inner])
    end
    inner = inner_network(peps, peps_prime)
    network_list = vcat(network_list, [inner])
    return network_list
end

# gradient of this function returns nothing.
@adjoint function generate_inner_network(
    peps::PEPS,
    peps_prime::PEPS,
    peps_prime_ham::PEPS,
    Hlocal::Array,
)
    adjoint_pullback(v) = (nothing, nothing, nothing, nothing)
    return generate_inner_network(peps, peps_prime, peps_prime_ham, Hlocal),
    adjoint_pullback
end

function rayleigh_quotient(inners::Array)
    self_inner = itensorad.scalar(inners[length(inners)])
    expectations = itensorad.scalar(sum(inners[1:length(inners)-1]))
    return expectations / self_inner
end

function loss_grad_wrap(peps::PEPS, Hlocal::Array)
    function loss(peps::PEPS)
        peps_prime = prime(peps; ham = false)
        peps_prime_ham = prime(peps; ham = true)
        network_list = generate_inner_network(peps, peps_prime, peps_prime_ham, Hlocal)
        variables = extract_data([peps, peps_prime, peps_prime_ham])
        inners = itensorad.batch_tensor_contraction(network_list, variables...)
        return rayleigh_quotient(inners)
    end
    loss_w_grad(peps::PEPS) = loss(peps), gradient(loss, peps)[1]
    return loss_w_grad
end

"""Update PEPS based on gradient descent
Parameters
----------
peps: a peps network with datatype PEPS
Hlocal: An array of MPO operators with datatype LocalMPO
stepsize: step size used in the gradient descent
num_sweeps: number of gradient descent sweeps/iterations
Returns
-------
An array containing Rayleigh quotient losses after each iteration.
"""
function gradient_descent(peps::PEPS, Hlocal::Array; stepsize::Float64, num_sweeps::Int)
    loss_w_grad = loss_grad_wrap(peps, Hlocal)
    # gradient descent iterations
    losses = []
    for iter = 1:num_sweeps
        l, g = loss_w_grad(peps)
        print("The rayleigh quotient at iteraton $iter is $l\n")
        peps = peps - stepsize * g
        push!(losses, l)
    end
    return losses
end

function optimize(peps::PEPS, Hlocal::Array; num_sweeps::Int, method = "GD")
    @assert(method in ["GD", "LBFGS", "CG"])
    inner(x, peps1, peps2) = peps1 * peps2
    loss_w_grad = loss_grad_wrap(peps, Hlocal)
    scale(peps, alpha) = alpha * peps
    add(peps1, peps2, alpha) = peps1 + alpha * peps2
    linesearch = HagerZhangLineSearch()
    if method == "GD"
        alg = GradientDescent(num_sweeps, 1e-8, linesearch, 2)
    elseif method == "LBFGS"
        alg = LBFGS(
            16;
            maxiter = num_sweeps,
            gradtol = 1e-8,
            linesearch = linesearch,
            verbosity = 2,
        )
    elseif method == "CG"
        alg = ConjugateGradient(;
            maxiter = num_sweeps,
            gradtol = 1e-8,
            linesearch = linesearch,
            verbosity = 2,
        )
    end
    _, _, _, _, history =
        OptimKit.optimize(loss_w_grad, peps, alg; inner = inner, scale! = scale, add! = add)
    return history[:, 1]
end
