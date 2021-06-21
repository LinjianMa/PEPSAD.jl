using ITensors
using AutoHOOT

const itensorad = AutoHOOT.ITensorsAD
const ad = AutoHOOT.autodiff
const gops = AutoHOOT.graphops

"""Generate an array of AutoHOOT nodes representing inner products, <p|H_1|p>, ..., <p|H_n|p>, <p|p>
Parameters
----------
peps: a peps network with datatype PEPS
Hlocal: An array of MPO operators with datatype LocalMPO
Returns
-------
An array of AutoHOOT nodes;
A dictionary mapping AutoHOOT input node to ITensor tensor
"""
function generate_inner_nodes(peps::PEPS, Hlocal::Array)
    network_list = []
    for H_term in Hlocal
        inner = inner_network(peps, H_term.mpo, [H_term.coord1, H_term.coord2])
        push!(network_list, inner)
    end
    push!(network_list, inner_network(peps))
    nodes, node_dict = itensorad.generate_einsum_expr(network_list)
    # TODO: add caching here
    for (i, n) in enumerate(nodes)
        nodes[i] = gops.generate_optimal_tree(n)
    end
    return nodes, node_dict
end

"""Compute rayleigh quotient gradient based on the input tensor network
inner products and their gradients.
"""
function rayleigh_quotient_gradient(inners::Array, grads_list::Array)
    #<p|p>
    self_inner = scalar(inners[length(inners)])
    #d(<p|p>)
    self_inner_grads = grads_list[length(inners)]
    #<p|H_i|p>
    inner_w_ham = inners[1:length(inners)-1]
    #d(<p|H_i|p>)
    inner_w_ham_grads = grads_list[1:length(inners)-1]

    updates = []
    update_size = length(grads_list[1])
    for u_index = 1:update_size
        # calculate the gradient of rayleigh quotient
        # for each term, equals d(<p|H_i|p>)/<p|p> - <p|H_i|p>d(<p|p>)/<p|p>^2
        update =
            inner_w_ham_grads[1][u_index] * (1.0 / self_inner) -
            inner_w_ham[1] * self_inner_grads[u_index] * (1.0 / self_inner / self_inner)
        for i = 2:length(inner_w_ham)
            update +=
                inner_w_ham_grads[i][u_index] * (1.0 / self_inner) -
                inner_w_ham[i] * self_inner_grads[u_index] * (1.0 / self_inner / self_inner)
        end
        push!(updates, update)
    end
    return updates
end

function rayleigh_quotient(inners::Array)
    self_inner = scalar(inners[length(inners)])
    return scalar(sum(inners[1:length(inners)-1]) / self_inner)
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
    inner_nodes, dict = generate_inner_nodes(peps, Hlocal)
    vec_peps = reshape(peps.data, peps.dimx * peps.dimy)
    innodes = [itensorad.retrieve_key(dict, t) for t in vec_peps]
    # build gradients
    gradients_list = []
    for inner in inner_nodes
        grads = ad.gradients(inner, innodes)
        push!(gradients_list, grads)
    end
    # gradient descent iterations
    losses = []
    for iter = 1:num_sweeps
        inner_tensors = itensorad.compute_graph(inner_nodes, dict)
        grads_tensors_list = []
        for grads in gradients_list
            grads_tensors = itensorad.compute_graph(grads, dict)
            push!(grads_tensors_list, grads_tensors)
        end
        # print out the Rayleigh quotient
        loss = rayleigh_quotient(inner_tensors)
        print("The rayleigh quotient at iteraton $iter is $loss\n")
        # gradient descent update
        rq_grads = rayleigh_quotient_gradient(inner_tensors, grads_tensors_list)
        for (j, node) in enumerate(innodes)
            dict[node] = dict[node] - stepsize * rq_grads[j]
        end
        push!(losses, loss)
    end
    return losses
end
