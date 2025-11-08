#!/bin/bash
#SBATCH -p debug
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -t 00:05:00

# 8 processos em 1 nó (cada nó suporta até 8 procs MPI)
mpirun -np 8 -N 8 ./knn_mpi nq=128 npp=400000 d=300 k=1024
