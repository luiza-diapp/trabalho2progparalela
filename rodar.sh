#!/bin/bash

# Este script executa o programa knn_mpi com os parâmetros especificados.

# O comando completo:
echo "Primeiro experimento np=1"
mpirun -np 1 ./knn_mpi nq=128 npp=400000 d=300 k=1024

echo "Segundo experimento np=8"
mpirun -np 8 ./knn_mpi nq=128 npp=400000 d=300 k=1024

echo "Terceiro experimento"
