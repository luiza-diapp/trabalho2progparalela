#include <stdio.h>

//ALUNOS:
//João Marcelo Caboclo - GRR20221227
//Luíza Diapp - GRR20221252

// Versão inicial (stub) — imprime a matriz R linearizada.
// O professor substituirá este arquivo por uma versão que verifica de verdade.
void verificaKNN( float *Q, int nq, float *P, int n, int D, int k, int *R ) {
    // note que R tem nq linhas por k colunas; acesso linear por [linha*k + coluna]
    printf("  ------------ VERIFICA KNN --------------- \n");
    for (int linha = 0; linha < nq; linha++)  {
        printf("knn[%d]: ", linha);
        for (int coluna = 0; coluna < k; coluna++)
            printf("%d ", R[ linha*k + coluna ]);
        printf("\n");
    }
}
