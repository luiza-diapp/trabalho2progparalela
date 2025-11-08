# //ALUNOS:
# //João Marcelo Caboclo - GRR20221227
# //Luíza Diapp - GRR20221252

# Compilador e flags
MPICC   = mpicc
CFLAGS  = -O3 -march=native -ffast-math -std=c11 -Wall -Wextra
LDFLAGS = -lm

# Nome do executável
TARGET = knn_mpi

# Arquivos fonte
SRCS = knn_mpi.c maxheap.c
HDRS = maxheap.h

# Regra padrão: compilar tudo
all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(MPICC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# Limpar arquivos compilados e saídas
clean:
	rm -f $(TARGET) *.o R.csv

