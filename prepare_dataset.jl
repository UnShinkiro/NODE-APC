using JLD
source_path = "../log_mels/"
save_path = "../preprocessed_data/clean-train-360/"
batch_size = 500

mel_folders = readdir(source_path)
file_count = 0
batch_count = 0
#lengths = []
all_mels = []
for folder in mel_folders
    sub_folders = readdir(string(source_path, folder))
    for sub_folder in sub_folders
        files = readdir(string(source_path, folder, "/", sub_folder))
        for file in files
            file_path = string(string(source_path, folder, "/", sub_folder, "/", file))
            log_mel = load(file_path)["log_mel"]
            #length = load(file_path)["length"]
            if file_count == 0
                all_mels = log_mel
                #append!(lengths, length)
                file_count += 1
                continue
            elseif file_count == batch_size
                println("Saving batch ", batch_count)
                save_name = save_path * "batch_$batch_count.jld"
                save(save_name, "all_mels", all_mels)
                sleep(5) # Allow time to save propaply 
                all_mels = []
                all_mels = log_mel
                batch_count += 1
                file_count = 1
            else
                all_mels = cat(all_mels, log_mel, dims=3)
                #append!(lengths, length)
                file_count += 1 
            end
        end
    end
end
if file_count < batch_size
    println("Saving batch ", batch_count)
    save_name = save_path * "batch_$batch_count.jld"
    save(save_name, "all_mels", all_mels)
end

batch_file_list = []
for batch = 0:batch_count
    batch_file_list = cat(batch_file_list, "batch_$batch.jld", dims=1)
end
save(save_path * "batch_file_list.jld", "file_list", batch_file_list)