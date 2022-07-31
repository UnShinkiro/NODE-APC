import Pkg
Pkg.add("FLAC")
Pkg.add("FileIO")
Pkg.add("DSP")
Pkg.add("FTTW")
Pkg.add("JLD")
using FLAC, FileIO, DSP, JLD
include("./zaf.jl")
using .zaf
#dataset_path = "LibriSpeech/dev-clean/"
dataset_path = "../LibriSpeech/train-clean-360/"
save_path = "log_mels/"
max_length = 800

speech_folders = readdir(dataset_path)
for folder in speech_folders
    sub_folders = readdir(string(dataset_path, folder))
    for sub_folder in sub_folders
        files = readdir(string(dataset_path, folder, "/", sub_folder))
        for file in files
            if occursin(".trans", file)
                println("skipping ", file)
            else
                file_path = string(string(dataset_path, folder, "/", sub_folder, "/", file))
                data, fs = load(file_path)
                window_length = nextpow(2, ceil(Int, 0.04*fs))
                window_function = zaf.hamming(window_length, "periodic")
                step_length = convert(Int, window_length/2)
                number_mels = 80
                mel_filterbank = zaf.melfilterbank(fs, window_length, number_mels)
                mel_spectrogram = zaf.melspectrogram(data, window_function, step_length, mel_filterbank)
                log_mel = log10.(mel_spectrogram)
                if size(log_mel)[2] < max_length
                    length = size(log_mel)[2]
                    log_mel = cat(log_mel, zeros(80, max_length - size(log_mel)[2]), dims=2)
                    log_mel = cat(log_mel, ones(80, 1) * length, dims = 2)
                else
                    length = max_length
                    log_mel = log_mel[:,1:max_length]
                    log_mel = cat(log_mel, ones(80, 1) * length, dims = 2)
                end
                log_mel = Float32.(log_mel)
                if any(isinf.(log_mel))
                    println("error: found inf in $file_path -- skipping")
                    continue
                elseif any(isnan.(log_mel))
                    println("error: found nan in $file_path -- skipping")
                    continue
                end
                file_name = string(save_path, folder, "/", sub_folder, "/", file, ".jld")
                save(file_name, "log_mel", log_mel)
            end
        end
    end
end