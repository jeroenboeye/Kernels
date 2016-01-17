#!/bin/sh

#PBS -N SUB_SHELL
#PBS -l walltime=11:59:00
#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -e suberror.file

module load scripts
module load Python/2.7.2-ictce-4.0.6
chmod 770 focal_model_swng.py
python focal_model_swng.py $LAM $MOR $BET $SHU $FRA
