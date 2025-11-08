#!/bin/bash

# Este script executa o programa knn_mpi com os parâmetros especificados.

# O comando completo:
mpirun -np 2 ./knn_mpi nq=128 npp=400000 d=300 k=1024