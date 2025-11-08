// knn_mpi.c
// Trabalho 2 - KNN com MPI + Max-Heap (decreaseMax via API)
// Compilar: mpicc -O3 -march=native -ffast-math -o knn_mpi knn_mpi.c maxheap.c -lm
// Executar: mpirun -np 8 ./knn_mpi nq=128 npp=400000 d=300 k=1024 [-v]

//ALUNOS:
//João Marcelo Caboclo - GRR20221227
//Luíza Diapp - GRR20221252

#define _POSIX_C_SOURCE 200112L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "maxheap.h"  
#include "verificaKNN.h"

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 64
#endif

//  Utils de parsing (nome=valor) e flag -v 
static int parse_int_kv(const char* s, const char* key, int* out) {
    const size_t len = strlen(key);
    if (strncmp(s, key, len) == 0 && s[len] == '=') {
        *out = atoi(s + len + 1);
        return 1;
    }
    return 0;
}

static void parse_args(int argc, char** argv,
                       int *nq, int *npp, int *d, int *k,
                       int *verify_flag) {
    *nq = 128; *npp = 400000; *d = 300; *k = 1024; *verify_flag = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) { *verify_flag = 1; continue; }
        if (parse_int_kv(argv[i], "nq", nq)) continue;
        if (parse_int_kv(argv[i], "npp", npp)) continue;
        if (parse_int_kv(argv[i], "n", npp))   continue; // sinônimo
        if (parse_int_kv(argv[i], "d", d))     continue;
        if (parse_int_kv(argv[i], "k", k))     continue;
    }
}

static void* xaligned_malloc(size_t nbytes) {
    // void *ptr = NULL;

    // #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    //     if (posix_memalign(&ptr, ALIGN_BYTES, nbytes) == 0) return ptr;
    // #elif __STDC_VERSION__ >= 201112L
    //     // C11 aligned_alloc requer que o tamanho seja múltiplo do alinhamento
    //     size_t sz = (nbytes + (ALIGN_BYTES - 1)) & ~(size_t)(ALIGN_BYTES - 1);
    //     ptr = aligned_alloc(ALIGN_BYTES, sz);
    //     if (ptr) return ptr;
    // #endif

    // Fallback universal
    return malloc(nbytes);
}

//  Geração de dados no rank 0 
static void geraConjuntoDeDados(float *C, int nc, int d, unsigned int seed) {
    // números em [-1, 1]
    srand(seed);
    long long total = (long long)nc * d;
    for (long long i = 0; i < total; i++) {
        C[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
    }
}

//  Distância ao quadrado 
static inline float dist2_sq(const float * __restrict__ q,
                             const float * __restrict__ p,
                             int d) {
    double acc = 0.0;
    for (int i = 0; i < d; i++) {
        double diff = (double)q[i] - (double)p[i];
        acc += diff * diff;
    }
    return (float)acc;
}

//  KNN para 1 consulta, usando MaxHeap 
static void knn_1query(const float *q, const float *P, int n, int d, int k,
                       int *out_idx) {
    if (k > n) k = n;

    MaxHeap h;
    maxheap_create(&h, k);

    // primeiros k pontos
    for (int j = 0; j < k; j++) {
        float d2 = dist2_sq(q, P + (long long)j * d, d);
        maxheap_push(&h, d2, j);
    }
    // restantes: decreaseMax implícito (push_or_decrease)
    for (int j = k; j < n; j++) {
        float d2 = dist2_sq(q, P + (long long)j * d, d);
        maxheap_push_or_decrease(&h, d2, j);
    }

    // saída ordenada por distância crescente (determinística p/ relatório)
    HeapItem *buf = (HeapItem*)malloc(sizeof(HeapItem) * k);
    if (!buf) {
        fprintf(stderr, "Erro: malloc buf k=%d\n", k);
        maxheap_destroy(&h);
        return; 
    }
    int got = maxheap_dump_sorted_ascending(&h, buf, k);
    for (int t = 0; t < got; ++t) out_idx[t] = buf[t].idx;
    free(buf);

    maxheap_destroy(&h);
}



//  Programa principal 
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nq, npp, d, k, verify_flag;
    parse_args(argc, argv, &nq, &npp, &d, &k, &verify_flag);

    if (rank == 0) {
        printf("Params: nq=%d npp=%d d=%d k=%d  procs=%d  verify=%s\n",
               nq, npp, d, k, nprocs, verify_flag ? "on" : "off");
        fflush(stdout);
    }

    // Para simplicidade: requer divisibilidade para Scatter/Gather simples
    if (nq % nprocs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Erro: nq (%d) deve ser divisivel por nprocs (%d) (use Scatterv/Gatherv se nao for).\n",
                    nq, nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 0;
    }
    const int local_nq = nq / nprocs;

    // Alocações
    float *P = (float*)xaligned_malloc(sizeof(float) * (long long)npp * d);
    float *Q = NULL; // só no root
    int   *R = NULL; // só no root (nq x k)
    float *Q_local = (float*)xaligned_malloc(sizeof(float) * (long long)local_nq * d);
    int   *R_local = (int*)  xaligned_malloc(sizeof(int)   * (long long)local_nq * k);

    if (!P || !Q_local || !R_local) {
        fprintf(stderr, "Rank %d: Falha ao alocar memoria.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (rank == 0) {
        Q = (float*)xaligned_malloc(sizeof(float) * (long long)nq * d);
        R = (int*)  xaligned_malloc(sizeof(int)   * (long long)nq * k);
        if (!Q || !R) {
            fprintf(stderr, "Root: Falha ao alocar Q/R.\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        // Gera dados apenas no root
        geraConjuntoDeDados(P, npp, d, 42u);
        geraConjuntoDeDados(Q, nq,  d, 4242u);
    }

    //  Cronometragem (inclui Tx P/Q, Compute, Rx R) 
    MPI_Barrier(MPI_COMM_WORLD); // ÚNICA barreira antes de começar a medir

    double t0 = MPI_Wtime();

    // Bcast de P
    double tb0 = MPI_Wtime();
    MPI_Bcast(P, (int)((long long)npp * d), MPI_FLOAT, 0, MPI_COMM_WORLD);
    double tb1 = MPI_Wtime();

    // Scatter de Q
    double ts0 = MPI_Wtime();
    MPI_Scatter(Q, (int)((long long)local_nq * d), MPI_FLOAT,
                Q_local, (int)((long long)local_nq * d), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();

    // Cálculo local do KNN
    double tc0 = MPI_Wtime();
    int k_eff = k > npp ? npp : k; //Garante que k nunca seja maior do que o número de pontos em P.

    for (int r = 0; r < local_nq; r++) {
        //Pega o endereço do vetor correspondente ao ponto r-ésimo de Q_local
        const float *q = Q_local + (long long)r * d; 
        //Calcula o endereço da linha de saída em R_local que vai guardar os k índices vizinhos para esse ponto q.
        int *out_row = R_local + (long long)r * k_eff;

        knn_1query(q, P, npp, d, k_eff, out_row);
    }
    double tc1 = MPI_Wtime();

    // Gather dos índices
    double tg0 = MPI_Wtime();
    MPI_Gather(R_local, (int)((long long)local_nq * k_eff), MPI_INT,
               R,        (int)((long long)local_nq * k_eff), MPI_INT,
               0, MPI_COMM_WORLD);
    double tg1 = MPI_Wtime();

    double t1 = MPI_Wtime();
    double t_total = t1 - t0;
    double t_bcast = tb1 - tb0;
    double t_scatt = ts1 - ts0;
    double t_comp  = tc1 - tc0;
    double t_gath  = tg1 - tg0;

    if (rank == 0) {

        printf("\n[KNN Timing]\n");
        printf("Total (incl. Tx P/Q, compute, Rx R): %.6f s\n", t_total);
        printf("  Bcast(P):   %.6f s\n", t_bcast);
        printf("  Scatter(Q): %.6f s\n", t_scatt);
        printf("  Compute:    %.6f s\n", t_comp);
        printf("  Gather(R):  %.6f s\n", t_gath);
        fflush(stdout);
    }

    //  Verificação (se -v), fora do tempo do KNN 
    if (verify_flag && rank == 0) {
        verificaKNN(Q, nq, P, npp, d, k_eff, R, 10);
    }


    if (rank == 0) { free(Q); free(R); }
    free(P); free(Q_local); free(R_local);

    MPI_Finalize();
    return 0;
}
