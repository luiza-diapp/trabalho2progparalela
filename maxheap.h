//ALUNOS:
//João Marcelo Caboclo - GRR20221227
//Luíza Diapp - GRR20221252

#define MAXHEAP_H
#include <stddef.h>

typedef struct {
    float key;   // distância^2
    int   idx;   // índice do ponto em P
} HeapItem;

typedef struct {
    HeapItem *data;   // vetor [capacity]
    int size;         // número de elementos no heap
    int capacity;     // capacidade máxima (k)
} MaxHeap;

// Cria um heap com capacidade "capacity". Retorna 0 em sucesso.
int  maxheap_create(MaxHeap *h, int capacity);

// Libera recursos (na prática, apenas zera ponteiros/contadores; não free externo).
void maxheap_destroy(MaxHeap *h);

// Insere um item quando ainda há espaço (< capacity). Retorna 1 se inseriu, 0 se não (capacidade cheia).
int  maxheap_push(MaxHeap *h, float key, int idx);

// Substitui a raiz por (key, idx) e restabelece o heap (decreaseMax). Retorna 1 se ok, 0 se heap vazio.
int  maxheap_decrease_max(MaxHeap *h, float key, int idx);

// Atalho útil no KNN: se ainda há espaço, insere; caso contrário, compara com a raiz e
// substitui (decreaseMax) SOMENTE se (key < raiz.key) ou empate por índice menor.
// Retorna: 1 se alterou o heap (inseriu/substituiu), 0 caso ignore.
int  maxheap_push_or_decrease(MaxHeap *h, float key, int idx);

// Remove todos os itens para um array ordenado por chave crescente (opcional para depuração/saída).
// Requer um buffer de tamanho >= h->size. Retorna o número de itens copiados.
int  maxheap_dump_sorted_ascending(MaxHeap *h, HeapItem *buffer, int max_items);

