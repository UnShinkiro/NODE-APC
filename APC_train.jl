# Starting Code on Julia version of APC
using Pkg
Pkg.activate("/home/z5195063/master/NODE-APC")
using Flux
using Flux.Losses: mae
using CUDA
using JLD
using BSON: @save
using IterTools: ncycle 
feature_size = 512
dataset_path = "../train-clean-360-jld/"

function build_APC_model(input_dimension)
    return Chain(
            LSTM(input_dimension, 512),
            Dropout(0.5),
            LSTM(512, 512),
            Dropout(0.5),
            LSTM(512, 512),
            Dropout(0.5),
            LSTM(512, 512))
end 


function build_postnet(output_dimension)
    return Dense(512 => output_dimension)
end


function getdata(file_path)
    data = load(file_path)["log_mel"]
    if (size(data)[1]) > 1600
        return data[1:1600]
    else
        return data
    end
end


function train()
    APC = build_APC_model(80) |> gpu
    post_net = build_postnet(80) |> gpu
    
    function loss(file)
        input = file[1:end-1] |> gpu
        output = file[2:end] |> gpu
        Flux.reset!(APC)
        #features = APC.(input) |> gpu
        #prediction = post_net.(features)[end] |>gpu
        features = [APC(cu(frame)) for frame in input] |> gpu
        prediction = [post_net(frame) for frame in features] |> gpu
        total_loss = sum(abs.(prediction .- output))
        println("batch size: ",size(file), "\tloss:", total_loss)
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
                train_loader = Flux.Data.DataLoader(batch_x, batchsize=1600, shuffle=false)
                Flux.train!(loss, Flux.params(APC, post_net), train_loader, ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
            end
        end
    end
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.0001))#, cb = Flux.throttle(evalcb, 30))
    return APC, post_net
end

trained_model, post_net = train()
trained_model = cpu(trained_model)
post_net = cpu(post_net)
@save "360hModel_v3.bson" trained_model post_net