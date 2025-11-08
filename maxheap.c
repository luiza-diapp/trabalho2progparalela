#include "maxheap.h"
#include <stdlib.h>
#include <string.h>

//ALUNOS:
//João Marcelo Caboclo - GRR20221227
//Luíza Diapp - GRR20221252

// typedef struct {
//     float key;   // distância^2
//     int   idx;   // índice do ponto em P
// } HeapItem;

// typedef struct {
//     HeapItem *data;   // vetor [capacity]
//     int size;         // número de elementos no heap
//     int capacity;     // capacidade máxima (k)
// } MaxHeap;

// Compara dois itens do heap, priorizando a chave e usando o índice para desempate.
static int item_greater(const HeapItem *a, const HeapItem *b) {
    if (a->key > b->key) return 1;
    if (a->key < b->key) return 0;
    return a->idx > b->idx; // tiebreak por índice maior = "maior" no max-heap
}


// Move um item para cima no heap para restaurar a propriedade Max-Heap após uma inserção.
static void heapify_up(MaxHeap *heap, int posicao) {
    HeapItem valor = heap->data[posicao];
    while (posicao > 0) {
        int pai = (posicao - 1) / 2;
        if (item_greater(&heap->data[pai], &valor)) break; // pai já é maior
        heap->data[posicao] = heap->data[pai];
        posicao = pai;
    }
    heap->data[posicao] = valor;
}

// Move um item para baixo no heap para restaurar a propriedade Max-Heap após uma remoção ou substituição.
static void heapify_down(MaxHeap *heap, int posicao) {
    int tamanho = heap->size;
    HeapItem valor = heap->data[posicao];
    while (1) {
        int filho_esq = 2 * posicao + 1;
        int filho_dir = filho_esq + 1;
        if (filho_esq >= tamanho) break; // sem filhos
        int maior_filho = filho_esq;         // maior filho
        if (filho_dir < tamanho && item_greater(&heap->data[filho_dir], &heap->data[filho_esq])) maior_filho = filho_dir;
        if (item_greater(&valor, &heap->data[maior_filho])) break; // já está ok
        heap->data[posicao] = heap->data[maior_filho];
        posicao = maior_filho;
    }
    heap->data[posicao] = valor;
}

// Inicializa um Max-Heap alocando memória para os dados.
int maxheap_create(MaxHeap *heap, int capacidade) {
    if (!heap || capacidade <= 0) return -1;
    heap->data = (HeapItem*)malloc((size_t)capacidade * sizeof(HeapItem));
    if (!heap->data) return -2;
    heap->size = 0;
    heap->capacity = capacidade;
    return 0;
}

// Libera a memória alocada para os dados do Max-Heap.
void maxheap_destroy(MaxHeap *heap) {
    if (!heap) return;
    free(heap->data);
    heap->data = NULL;
    heap->size = 0;
    heap->capacity = 0;
}

// Insere um novo item no heap e o reposiciona (heapify-up) para manter a ordem.
int maxheap_push(MaxHeap *heap, float chave, int indice) {
    if (!heap || heap->size >= heap->capacity) return 0;
    int posicao = heap->size++;
    heap->data[posicao].key = chave;
    heap->data[posicao].idx = indice;
    heapify_up(heap, posicao);
    return 1;
}

// Substitui o item máximo (raiz) por um novo item com valor **menor** e restaura a ordem (heapify-down).
int maxheap_decrease_max(MaxHeap *heap, float chave, int indice) {
    if (!heap || heap->size == 0) return 0;
    heap->data[0].key = chave;
    heap->data[0].idx = indice;
    heapify_down(heap, 0);
    return 1;
}

// Insere um item ou substitui o máximo se o novo item tiver uma chave "melhor" (menor), mantendo o tamanho.
int maxheap_push_or_decrease(MaxHeap *heap, float chave, int indice) {
    if (!heap) return 0;
    if (heap->size < heap->capacity) {
        return maxheap_push(heap, chave, indice);
    }
    // Cheio: compara com a raiz (maior dos k atuais)
    HeapItem topo = heap->data[0];
    int substituir = 0;
    // Se key for melhor (menor) que a raiz, substitui
    if (chave < topo.key) substituir = 1;
    else if (chave == topo.key && indice < topo.idx) substituir = 1; // desempate por idx menor

    if (substituir) return maxheap_decrease_max(heap, chave, indice);
    return 0; // ignorado
}

// Remove e retorna o item com a chave máxima (raiz), restaurando a ordem do heap (heapify-down).
static int pop_max(MaxHeap *heap, HeapItem *saida) {
    if (heap->size == 0) return 0;
    if (saida) *saida = heap->data[0];
    heap->data[0] = heap->data[heap->size - 1];
    heap->size--;
    if (heap->size > 0) heapify_down(heap, 0);
    return 1;
}

// Extrai todos os itens do heap, classificando-os em ordem **crescente** de chave (do menor para o maior).
int maxheap_dump_sorted_ascending(MaxHeap *heap, HeapItem *buffer, int max_itens) {
    if (!heap || !buffer || max_itens <= 0) return 0;
    // Faz uma cópia para não destruir o heap do chamador
    MaxHeap temporario = *heap;
    temporario.data = (HeapItem*)malloc((size_t)heap->size * sizeof(HeapItem));
    if (!temporario.data) return 0;
    memcpy(temporario.data, heap->data, (size_t)heap->size * sizeof(HeapItem));

    int contagem_saida = 0;
    // Extrair em ordem decrescente e reverter
    HeapItem *reverso = (HeapItem*)malloc((size_t)heap->size * sizeof(HeapItem));
    if (!reverso) { free(temporario.data); return 0; }

    while (temporario.size > 0 && contagem_saida < max_itens) {
        pop_max(&temporario, &reverso[contagem_saida++]);
    }
    // rev[0..out_count-1] está em ordem decrescente; copia invertido para buffer
    for (int i = 0; i < contagem_saida; ++i) buffer[i] = reverso[contagem_saida - 1 - i];

    free(reverso);
    free(temporario.data);
    return contagem_saida;
}