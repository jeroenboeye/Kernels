#!/bin/sh

#PBS -N MAIN_SHELL
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -o outputmain.file
#PBS -e errormain.file

growthrates=(2)
mort=(0 0.005 0.01 0.05 0.1 0.2 0.5 0.7)
beta=(1)
shuffle=(1)
fraction=(0 1 2 3 4 5 6 7 8 9 10 11)


for L in "${growthrates[@]}"
do
for M in "${mort[@]}"
do
for B in "${beta[@]}"
do
for S in "${shuffle[@]}"
do
for F in "${fraction[@]}"
do

export LAM=$L
export MOR=$M
export BET=$B
export SHU=$S
export FRA=$F
qsub sub_shell_swng.sh -v LAM,MOR,BET,SHU,FRA -t 1-10
done
done
done
done
done