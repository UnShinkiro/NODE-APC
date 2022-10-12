# Starting Code on Julia version of APC
using Pkg
Pkg.activate(".")
using Flux
using Flux.Losses: mae
using CUDA
using JLD
using BSON: @save
using IterTools: ncycle 
batch_size = 50
max_len = 800
feature_size = 512
dataset_path = "../preprocessed_data/clean-train-360/"

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


function getdata(batch_file_path)
    data = load(batch_file_path)["all_mels"]
    return [[data[:,frame,idx] for frame=1:Int(data[1,end,idx])] for idx=1:size(data)[3]]
end


function train()
    APC = build_APC_model(80) |> gpu
    post_net = build_postnet(80) |> gpu
    
    function loss(batch_data)
        total_loss = 0
        for file in batch_data
            input = file[1:end-1] |> gpu
            output = file[end] |> gpu
            Flux.reset!(APC)
            Flux.reset!(post_net)
            features = APC.(input) |> gpu
            prediction = post_net.(features)[end] |> gpu
            total_loss += sum(abs.(prediction .- output))
        end
        println("batch size: ",size(batch_data), "\tloss:", total_loss)
        return total_loss
    end
    
    opt = ADAM(0.01)
    #tx, tl = (batch_x[:,:,5], batch_x[:,max_len+1,5])
    #evalcb = () -> @show println("Next epoch")
    
    file_list = load(dataset_path * "batch_file_list.jld")["file_list"]
    validation = getdata(dataset_path * "batch_0.jld")
    for epoch = 1:1
        for file_name in file_list
            file_path = dataset_path * file_name
            if file_name == "batch_0.jld"
                continue
            else
                batch_x = getdata(file_path)
                println(size(batch_x))
                println("training using $file_name")
                train_loader = Flux.Data.DataLoader(batch_x, batchsize=50, shuffle=true)
                Flux.train!(loss, Flux.params(APC, post_net), train_loader, ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
                println("validation loss:\n", loss(validation))
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
@save "360hModel.bson" trained_model post_net
