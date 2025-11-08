#include <stdio.h>
#include "maxheap.h"

//ALUNOS:
//João Marcelo Caboclo - GRR20221227
//Luíza Diapp - GRR20221252

// Versão inicial (stub) — imprime a matriz R linearizada.
// O professor substituirá este arquivo por uma versão que verifica de verdade.

// void verificaKNN( float *Q, int nq, float *P, int n, int D, int k, int *R ) {
//     // note que R tem nq linhas por k colunas; acesso linear por [linha*k + coluna]
//     printf("  ------------ VERIFICA KNN --------------- \n");
//     for (int linha = 0; linha < nq; linha++)  {
//         printf("knn[%d]: ", linha);
//         for (int coluna = 0; coluna < k; coluna++)
//             printf("%d ", R[ linha*k + coluna ]);
//         printf("\n");
//     }
// }

int cmp_HeapItem_asc_index_tiebreak(const void *a, const void *b) {
    const HeapItem *pa = (const HeapItem*)a;
    const HeapItem *pb = (const HeapItem*)b;
    if (pa->key < pb->key) return -1;
    if (pa->key > pb->key) return  1;
    return (pa->idx > pb->idx) - (pa->idx < pb->idx);
}

/* Verificador REAL (recalcula k-NN e compara com R) — chame só com -v */
void verificaKNN(const float *Q, int nq, float *P, int n, int d, int k, int *R, int L /*linhas a verificar*/) {
    if (k > n) k = n;
    if (L > nq) L = nq;
    printf("\n[Verificacao REAL] Conferindo %d linhas com k=%d ...\n", L, k);

    /* buffers auxiliares por linha */
    float *hD = (float*)malloc(sizeof(float) * k);
    int   *hI = (int*)  malloc(sizeof(int)   * k);
    HeapItem  *gt  = (HeapItem*)malloc(sizeof(HeapItem)  * k);

    if (!hD || !hI || !gt) {
        fprintf(stderr, "Falha ao alocar buffers da verificacao.\n");
        free(hD); free(hI); free(gt);
        return;
    }

    int erros = 0;

    for (int r = 0; r < L; r++) {
        const float *q = Q + (long long)r * d;

        /* 1) inicializa heap (k primeiros) */
        for (int j = 0; j < k; j++) {
            double acc = 0.0;
            const float *pj = P + (long long)j * d;
            for (int t = 0; t < d; t++) {
                double diff = (double)q[t] - (double)pj[t];
                acc += diff * diff;
            }
            hD[j] = (float)acc;
            hI[j] = j;
        }
        /* build max-heap (método de Floyd) */
        for (int i = (k >> 1) - 1; i >= 0; i--) {
            int x = i;
            while (1) {
                int l = 2*x + 1, rr = 2*x + 2, m = x;
                if (l  < k && hD[l]  > hD[m]) m = l;
                if (rr < k && hD[rr] > hD[m]) m = rr;
                if (m != x) {
                    float td = hD[x]; hD[x] = hD[m]; hD[m] = td;
                    int   ti = hI[x]; hI[x] = hI[m]; hI[m] = ti;
                    x = m;
                } else break;
            }
        }

        /* 2) varre o resto de P mantendo k melhores (decreaseMax implícito) */
        for (int j = k; j < n; j++) {
            double acc = 0.0;
            const float *pj = P + (long long)j * d;
            for (int t = 0; t < d; t++) {
                double diff = (double)q[t] - (double)pj[t];
                acc += diff * diff;
            }
            float d2 = (float)acc;
            if (d2 < hD[0]) {
                hD[0] = d2;
                hI[0] = j;
                /* heapify_down na raiz */
                int x = 0;
                while (1) {
                    int l = 2*x + 1, rr = 2*x + 2, m = x;
                    if (l  < k && hD[l]  > hD[m]) m = l;
                    if (rr < k && hD[rr] > hD[m]) m = rr;
                    if (m != x) {
                        float td = hD[x]; hD[x] = hD[m]; hD[m] = td;
                        int   ti = hI[x]; hI[x] = hI[m]; hI[m] = ti;
                        x = m;
                    } else break;
                }
            }
        }

        /* 3) ordena ground truth por distância crescente (determinístico) */
        for (int t = 0; t < k; t++) { gt[t].key = hD[t]; gt[t].idx = hI[t]; }
        qsort(gt, k, sizeof(HeapItem), cmp_HeapItem_asc_index_tiebreak);

        /* 4) validações */
        const int *row = R + (long long)r * k;
        int ok = 1;

        /* índices válidos e não-decréscimo das distâncias */
        for (int t = 0; t < k; t++) {
            if (row[t] < 0 || row[t] >= n) { ok = 0; break; }
            if (t > 0) {
                double dprev = 0.0, dcurr = 0.0;
                const float *pprev = P + (long long)row[t-1] * d;
                const float *pcurr = P + (long long)row[t]   * d;
                for (int u = 0; u < d; u++) {
                    double dp = (double)q[u] - (double)pprev[u];
                    double dc = (double)q[u] - (double)pcurr[u];
                    dprev += dp*dp; dcurr += dc*dc;
                }
                if (dprev > dcurr + 1e-6) ok = 0;
            }
        }

        /* compara com ground truth (índices e ordem) */
        for (int t = 0; t < k && ok; t++) {
            if (row[t] != gt[t].idx) ok = 0;
        }

        if (!ok) {
            erros++;
            printf("Linha %d: MISMATCH\n", r);
            printf("  R:  "); for (int t = 0; t < k; t++) printf("%d ", row[t]);
            printf("\n  GT: "); for (int t = 0; t < k; t++) printf("%d ", gt[t].idx);
            printf("\n");
        } else {
            printf("Linha %d: OK\n", r);
        }
    }

    if (erros == 0) printf(">> Verificacao REAL: tudo OK nas %d linhas.\n", L);
    else            printf(">> Verificacao REAL: %d linhas com divergencia.\n", erros);

    free(hD); free(hI); free(gt);
}
