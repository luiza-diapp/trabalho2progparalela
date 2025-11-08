// verificaKNN.c — versão reescrita usando apenas a API de max-heap do projeto.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "maxheap.h"
#include "verificaKNN.h"

static inline float dist2_sq(const float *q, const float *p, int d) {
    double acc = 0.0;
    for (int i = 0; i < d; i++) {
        double diff = (double)q[i] - (double)p[i];
        acc += diff * diff;
    }
    return (float)acc;
}

// Ordenação final determinística (distância ↑, desempate por índice ↑)
static int cmp_asc_key_then_idx(const void *a, const void *b) {
    const HeapItem *x = (const HeapItem*)a;
    const HeapItem *y = (const HeapItem*)b;
    if (x->key < y->key) return -1;
    if (x->key > y->key) return  1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

void verificaKNN(const float *Q, int nq, float *P, int n, int d, int k, int *R, int L)
{
    if (k > n) k = n;
    if (L > nq) L = nq;

    printf("\n[Verificacao] Conferindo %d linhas com k=%d …\n", L, k);

    // buffers auxiliares
    HeapItem *gt = (HeapItem*)malloc((size_t)k * sizeof(HeapItem));
    if (!gt) {
        fprintf(stderr, "Falha ao alocar buffer da verificacao.\n");
        return;
    }

    int linhas_com_erro = 0;
    const double EPS = 1e-7;

    for (int r = 0; r < L; r++) {
        const float *q = Q + (long long)r * d;

        // 1) constrói verdade-terra por força bruta usando a API do MaxHeap
        MaxHeap h;
        maxheap_create(&h, k);

        for (int j = 0; j < n; j++) {
            float d2 = dist2_sq(q, P + (long long)j * d, d);

            if (h.size < h.capacity) {
                maxheap_push(&h, d2, j);
            } else {
                // mantém os k menores por (dist, idx)
                const HeapItem *root = &h.data[0];
                if (d2 < root->key || (d2 == root->key && j < root->idx)) {
                    maxheap_decrease_max(&h, d2, j);
                }
            }
        }

        // Dump em ordem crescente (dist, idx) para comparar com R
        int got = maxheap_dump_sorted_ascending(&h, gt, k);
        if (got != k) {
            // fallback: copia manualmente e ordena (não deveria acontecer)
            for (int i = 0; i < h.size && i < k; ++i) gt[i] = h.data[i];
            qsort(gt, h.size, sizeof(HeapItem), cmp_asc_key_then_idx);
        }
        maxheap_destroy(&h);

        // 2) validações da linha retornada
        const int *row = R + (long long)r * k;
        int ok = 1;

        // 2.1) índices válidos
        for (int t = 0; t < k; t++) {
            if (row[t] < 0 || row[t] >= n) 
            { ok = 0; printf("\n1\n");break; }
        }

        // 2.2) distâncias não-decrecentes em R
        if (ok) {
            for (int t = 1; t < k; t++) {
                double dprev = 0.0, dcurr = 0.0;
                const float *pprev = P + (long long)row[t-1] * d;
                const float *pcurr = P + (long long)row[t]   * d;
                for (int u = 0; u < d; u++) {
                    double dp = (double)q[u] - (double)pprev[u];
                    double dc = (double)q[u] - (double)pcurr[u];
                    dprev += dp*dp; dcurr += dc*dc;
                }
                if (fabs(dprev - dcurr) < EPS) { 
                    ok = 0; 
                    printf("\n2\n");
                    break; }
            }
        }

        // 2.3) igualdade com a verdade-terra (mesma ordem)
        if (ok) {
            for (int t = 0; t < k; t++) {
                if (row[t] != gt[t].idx) 
                { ok = 0; printf("\n3\n"); break; }
            }
        }

        if (!ok) {
            linhas_com_erro++;
            printf("Linha %d: MISMATCH\n", r);

            // mostra sequência R e GT
            printf("  R : ");
            for (int t = 0; t < k; t++) printf("%d ", row[t]);
            printf("\n  GT: ");
            for (int t = 0; t < k; t++) printf("%d ", gt[t].idx);
            printf("\n");

            // contexto na primeira divergência
            int pos = -1;
            for (int t = 0; t < k; ++t) { if (row[t] != gt[t].idx) { pos = t; break; } }
            if (pos >= 0) {
                int a = (pos - 5 < 0) ? 0 : pos - 5;
                int b = (pos + 5 >= k) ? k - 1 : pos + 5;
                printf("  Contexto [%d..%d]:\n", a, b);
                for (int t = a; t <= b; ++t) {
                    int ir = row[t];
                    int ig = gt[t].idx;
                    double dr = 0.0, dg = 0.0;
                    const float *pr = P + (long long)ir * d;
                    const float *pg = P + (long long)ig * d;
                    for (int u = 0; u < d; ++u) {
                        double rr = (double)q[u] - (double)pr[u];
                        double gg = (double)q[u] - (double)pg[u];
                        dr += rr*rr; dg += gg*gg;
                    }
                    printf("    t=%d: R[%d]=%d d2=%.9g | GT[%d]=%d d2=%.9g\n",
                           t, t, ir, dr, t, ig, dg);
                }
            }
        } else {
            printf("Linha %d: OK\n", r);
        }
    }

    if (linhas_com_erro == 0)
        printf(">> Verificacao: tudo OK nas %d linhas.\n", L);
    else
        printf(">> Verificacao: %d linha(s) com divergencia.\n", linhas_com_erro);

    free(gt);
}
