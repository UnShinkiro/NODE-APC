# Starting Code on Julia version of APC
using Pkg
Pkg.activate(".")
using Flux
using Flux.Losses: mae
using CUDA
using JLD
using DiffEqFlux
using DifferentialEquations
using BSON: @save
using IterTools: ncycle 
batch_size = 50
max_len = 1200
feature_size = 512
dataset_path = "../dev-clean-jld/"

function build_node_apc(prenet, node)
    return Chain(
        prenet,
        node
    )
end 

function build_postnet(feature_dimension, mel_dimension)
    return Dense(feature_dimension => mel_dimension)
end

function build_prenet(mel_dimension, feature_dimension)
    return Dense(mel_dimension, feature_dimension)
end

function build_neural_layer(feature_dimension)
    return LSTM(feature_dimension, feature_dimension)
end

lspan = (0.0f0,1.0f0)
# l = range(lspan[1],lspan[2],length=4)

function build_node(neural_layer)
    return NeuralODE(neural_layer,lspan,Tsit5(),save_start=false,saveat=1,reltol=1e-7,abstol=1e-9)
end
# save_start = false

function getdata(file_path)
    data = load(file_path)["log_mel"]
    if (size(data)[1]) > 1200
        return data[1:1200]
    else
        return data
    end
end

function train()
    prenet = build_prenet(80, 512) |> gpu
    # save this after training so can rebuild nerual_ode with trained weights but have more evaluation steps
    neural_layer = build_neural_layer(512) |> gpu
    node = build_node(neural_layer) |> gpu
    post_net = build_postnet(512, 80) |> gpu
    node_apc = build_node_apc(prenet, node) |> gpu
    
    function loss(file)
        input = file[1:end-1] |> gpu
        output = file[end] |> gpu
        Flux.reset!(node_apc)
        features = node_apc.(input) |> gpu
        prediction = post_net.(features)[end] |> gpu
        total_loss = sum(abs.(prediction .- output))
        println("loss:", total_loss)
        return total_loss
    end
    
    opt = ADAM(0.01)
    #tx, tl = (batch_x[:,:,5], batch_x[:,max_len+1,5])
    #evalcb = () -> @show println("Next epoch")
    
    file_list = readdir(dataset_path)
    file_count = 0
    for epoch = 1:1
        for file_name in file_list
            file_count += 1
            file_path = dataset_path * file_name
            if file_name == "batch_0.jld"
                continue
            else
                batch_x = getdata(file_path)
                println("training using $file_name $file_count")
                train_loader = Flux.Data.DataLoader(batch_x, batchsize=1200, shuffle=false)
                Flux.train!(loss, Flux.params(node_apc, post_net), train_loader, ADAM(0.0001))#, cb = Flux.throttle(evalcb, 30))
            end
        end
    end
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.0001))#, cb = Flux.throttle(evalcb, 30))
    return neural_layer, post_net
end

trained_model, post_net = train()
trained_model = cpu(trained_model)
post_net = cpu(post_net)
@save "360hModel_node-apc.bson" trained_model post_net