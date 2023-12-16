#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


//rows = vector size
//#define MATRIX_SIZE 4
#define MATRIX_ROWS 30
#define MATRIX_COLS 4
#define VECTOR_SIZE 30

void printVector(int* vector, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", vector[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;

    int matrix[MATRIX_ROWS][MATRIX_COLS] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {13, 14, 15, 16},
        {13, 14, 15, 16},
        {13, 14, 15, 25}
    };

    int vector[VECTOR_SIZE] = { 1, 2, 3, 4, 5 };
    int result[VECTOR_SIZE] = { 0 };

    start_time = MPI_Wtime();  // Замеряем начальное время

    int rowsPerProcess = MATRIX_ROWS / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    int* gatheredResult = NULL;

    if (rank == 0) {
        gatheredResult = (int*)malloc(VECTOR_SIZE * sizeof(int));
    }

    MPI_Gather(result + startRow, rowsPerProcess, MPI_INT, gatheredResult, rowsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();  // Замеряем конечное время

    if (rank == 0) {
        printf("Matrix:\n");
        for (int i = 0; i < MATRIX_ROWS; i++) {
            for (int j = 0; j < MATRIX_COLS; j++) {
                printf("%d\t", matrix[i][j]);
            }
            printf("\n");
        }

        printf("\nVector:\n");
        printVector(vector, VECTOR_SIZE);

        printf("\nResult:\n");
        printVector(gatheredResult, VECTOR_SIZE);

        printf("\nElapsed Time: %f seconds\n", end_time - start_time);

        free(gatheredResult);
    }

    MPI_Finalize();
    return 0;
}