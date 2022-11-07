using JLD
source_path = "../train-clean-360-jld/"
save_path = "../preprocessed_data/clean-train-360/"
batch_size = 32

files = readdir(source_path)
file_count = 1
batch_count = 1
for file in files
    println("Handling file ", file)
    log_mel = load(source_path * file)["log_mel"]
    if file_count == 1
        batch_mel = log_mel
    else
        if file_count == batch_size
            file_count = 1
            println("Saving batch ", batch_count)
            save_name = save_path * "batch_$batch_count.jld"
            save(save_name, "batch_mel", batch_mel)
            batch_count += 1
        end
        batch_mel = [batch_mel log_mel]
    end
end
if file_count < batch_size
    println("Saving batch ", batch_count)
    save_name = save_path * "batch_$batch_count.jld"
    save(save_name, "batch_mel", batch_mel)
end