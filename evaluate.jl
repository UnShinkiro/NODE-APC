using Pkg
Pkg.activate(".")
using Flux
using CUDA
using JLD
using DiffEqFlux
using DifferentialEquations
using BSON: @load
using IterTools: ncycle 
global APC
global post_net
dataset_path = "../train-clean-360-jld/"
using_NODE = true

if using_NODE
    @load "/srv/scratch/z5195063/devNODEModel.bson" prenet trained_model post_net
    lspan = (0.0f0,1.0f0)
    node = NeuralODE(neural_layer,lspan,Tsit5(),save_start=false,saveat=1,reltol=1e-7,abstol=1e-9)
    APC = Chain(prenet, node) |> gpu
    post_net = post_net |> gpu
else
    @load "/srv/scratch/z5195063/360hModel_v3.bson" trained_model post_net
    APC = trained_model |> gpu
    post_net = post_net |> gpu
end

function loss(file)
    input = file[1:end-1] |> gpu
    output = file[2:end] |> gpu
    Flux.reset!(APC)
    #features = APC.(input) |> gpu
    #prediction = post_net.(features)[end] |>gpu
    features = [APC(cu(frame)) for frame in input] |> gpu
    prediction = [post_net(frame) for frame in features] |> gpu
    total_loss = sum([sum(abs.(prediction[idx] .- output[idx])) for idx=1:size(output)[1]])/size(output)[1]
    println("batch size: ",size(file), "\tloss:", total_loss)
    return total_loss
end

file_list = readdir(dataset_path)
file_count = 0
for file_name in file_list
    if file_count > 3000
        break
    end
    file_count += 1
    file_path = dataset_path * file_name
    loss(file)
end

