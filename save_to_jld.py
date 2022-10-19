from julia import Main
Main.eval('using Pkg; Pkg.activate(".")')
Main.eval('using JLD')

file_list = []
filepath = "train-clean-360/"
file = open("train-clean-360", "r")

for line in file:
    data = line.strip().split()
    if len(data) == 1:
        if data[0] == '.':  # end of the current utterance
            Main.filename = f'{filepath}/{utt_id}.jld'
            Main.data = log_mel
            Main.eval('data = Float32.(data)')
            Main.eval('data = [data[frame,:] for frame=1:size(data)[1]]')
            Main.eval('save(filename, "log_mel", data)')

        else: # here starts a new utterance
            utt_id = data[0]
            print(f'processing{utt_id}')
            log_mel = []
            file_list.append(f'{utt_id}.jld')

    else:
        log_mel.append([float(i) for i in data])

Main.fileList = file_list
Main.eval('save("dev/fileList.jld", "fileList", fileList)')