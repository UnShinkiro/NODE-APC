#!/bin/bash
 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P wa66
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/wa66
#PBS -l wd

cd $PBS_JOBFS
cp /g/data/wa66/Tong/dev-clean.tar.gz .
cp /g/data/wa66/Tong/test-clean.tar.gz .
cp /g/data/wa66/Tong/train-clean-100.tar.gz .
tar -xf dev-clean.tar.gz 
tar -xf test-clean.tar.gz
tar -xf train-clean-100.tar.gz
cp /g/data/wa66/Tong/change_ymal.sh LibriSpeech/
cd LibriSpeech
bash change_ymal.sh
module load python3/3.8.5
module load julia/1.7.1
module load cuda/10.1

cd $HOME/s3prl/s3prl
pip install -e ..
python3 run_downstream.py -n CUSTOM_CTC_FINAL -m train -u customized_upstream -d ctc -c downstream/ctc/libriphone.yaml > $HOME/basline_benchmark.txt
