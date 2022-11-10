#!/bin/bash
 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P wa66
#PBS -l walltime=2:00:00
#PBS -l storage=gdata/wa66
#PBS -l wd

module load julia/1.7.1
module load cuda/10.1
cd $PBS_JOBFS
cp -r $HOME/NODE-APC .
cp /g/data/wa66/Tong/devDownscaledAPCmodel.bson .
cp /g/data/wa66/Tong/train-clean-360-jld.tar.gz .
tar -xf train-clean-360-jld.tar.gz
cd NODE-APC
julia evaluate.jl downscale > $HOME/devDownscaledmodel.txt
