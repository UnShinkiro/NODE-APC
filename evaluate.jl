using Pkg
Pkg.activate(".")
using Flux
using CUDA
using JLD
using DiffEqFlux
using Random
using DifferentialEquations
using BSON: @load
using IterTools: ncycle 
dataset_path = "../train-clean-360-jld/"
using_model = ARGS[1]

function getdata(file_path)
    data = load(file_path)["log_mel"]
    if (size(data)[1]) > 1600
        return data[1:1600]
    else
        return data
    end
end

function evaluate()
    file_list = readdir(dataset_path)
    file_count = 0

    if using_model == "node"
        print("using NODE-APC")
        @load "../devNODEmodel.bson" prenet trained_model post_net
        lspan = (0.0f0,1.0f0)
        prenet = prenet |> gpu
        post_net = post_net |> gpu
        trained_model = trained_model |> gpu
        node = NeuralODE(trained_model,lspan,Tsit5(),save_start=false,saveat=1,reltol=1e-7,abstol=1e-9) |> gpu
        APC = Chain(prenet, node) |> gpu
    elseif using_model == "dev"
        print("using dev trained full")
        @load "../devAPCmodel.bson" trained_model post_net
        APC = trained_model |> gpu
        post_net = post_net |> gpu
    elseif using_model == "downscale"
        print("using dev trained downscale")
        @load "../devDownscaledAPCmodel" trained_model post_net
        APC = trained_model |> gpu
        post_net = post_net |> gpu
    else
        print("using 360hTrained full")
        @load "../360hModel_v3.bson" trained_model post_net
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

    for file_name in file_list
        if file_count > 3000
            break
        end
        file_count += 1
        file_path = dataset_path * file_name
        file = getdata(file_path)
        loss(file)
    end
end

evaluate()
