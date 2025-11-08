#!/bin/bash
#SBATCH -p debug
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -t 00:05:00

mpirun -np 1 -N 1 ./knn_mpi nq=128 npp=400000 d=300 k=1024
