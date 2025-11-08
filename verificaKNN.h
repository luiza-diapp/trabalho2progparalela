#ifndef VERIFICA_KNN_H
#define VERIFICA_KNN_H

#include <stddef.h>

// Verifica as k vizinhanças retornadas em R comparando com o gabarito exato (força bruta).
// Q: consultas [nq x d] (row-major), P: base [n x d], k: vizinhos, R: índices [nq x k].
// L: número de linhas (consultas) a verificar (as primeiras L).
void verificaKNN(const float *Q, int nq, float *P, int n, int d, int k, int *R, int L);

#endif // VERIFICA_KNN_H