# Starting Code on Julia version of APC
using Pkg
Pkg.activate(".")
using Flux
using Flux.Losses: mae
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
    all_mels = load(batch_file_path)["all_mels"]
    #lengths = load("100_speech.jld")["lengths"]
    
    return all_mels
end


function train()
    APC = build_APC_model(80)
    post_net = build_postnet(80)
    
    function loss(batch_data)
        total_loss = 0
        for idx=1:size(batch_data)[3]
            data = batch_data[:,:,idx]
            length = Int(data[1,end])
            input = data[:, 1:length-1]
            output = data[:, length]
            Flux.reset!(APC)
            Flux.reset!(post_net)
            features = APC(input)
            #print(size(features))
            prediction = post_net(features[:,end])
            total_loss += sum(abs.(prediction.-output))
        end
        println("batch size: ",size(batch_data), "\tloss:", total_loss)
        return total_loss
    end
    
    opt = ADAM(0.01)
    #tx, tl = (batch_x[:,:,5], batch_x[:,max_len+1,5])
    #evalcb = () -> @show println("Next epoch")
    
    file_list = load(dataset_path * "batch_file_list.jld")["file_list"]
    for epoch = 1:5
        for file_name in file_list
            #file_name = "dev_speech.jld"
            file_path = dataset_path * file_name
            batch_x = getdata(file_path)
            println(size(batch_x))
            println("training using $file_name")
            train_loader = Flux.Data.DataLoader(batch_x, batchsize=50, shuffle=true)
            Flux.train!(loss, Flux.params(APC, post_net), train_loader, ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
        end
    end
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.001))#, cb = Flux.throttle(evalcb, 30))
    #Flux.train!(loss, Flux.params(APC, post_net), ncycle(train_loader, 10), ADAM(0.0001))#, cb = Flux.throttle(evalcb, 30))
    return APC, post_net
end

#trained_model, post_net = train()
trained_model = ""
post_net = ""
@save "360hModel.bson" trained_model post_net
