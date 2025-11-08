#!/bin/bash
#SBATCH -p debug
#SBATCH --exclusive
#SBATCH -N 4
#SBATCH -t 00:05:00

# 8 processos no total, 4 nós, 2 por nó
mpirun -np 8 -N 2 ./knn_mpi nq=128 npp=400000 d=300 k=1024
