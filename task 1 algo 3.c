#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 4
#define VECTOR_SIZE 4

void printVector(int *vector, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", vector[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d\t", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void matrixVectorMultiply(int *matrix, int *vector, int *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = 0;
        for (int j = 0; j < size; j++) {
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
}

void parallelMatrixVectorMultiply(int *matrix, int *vector, int *result, int size, int rank, int num_procs) {
    int block_size = size / num_procs;
    int *local_result = (int *)malloc(block_size * sizeof(int));

    // Scatter matrix rows to all processes
    MPI_Scatter(matrix, block_size * size, MPI_INT, MPI_IN_PLACE, block_size * size, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast vector to all processes
    MPI_Bcast(vector, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local result
    matrixVectorMultiply(matrix, vector, local_result, block_size);

    // Gather local results to the root process
    MPI_Gather(local_result, block_size, MPI_INT, result, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_result);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;

    int matrix[MATRIX_SIZE][MATRIX_SIZE] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int vector[VECTOR_SIZE] = {1, 2, 3, 4};
    int result[MATRIX_SIZE] = {0};

    if (rank == 0) {
        printf("Matrix:\n");
        printMatrix((int *)matrix, MATRIX_SIZE);
        printf("\nVector:\n");
        printVector(vector, VECTOR_SIZE);
    }

    start_time = MPI_Wtime(); // Measure start time

    parallelMatrixVectorMultiply((int *)matrix, vector, result, MATRIX_SIZE, rank, size);

    end_time = MPI_Wtime(); // Measure end time

    if (rank == 0) {
        printf("\nResult:\n");
        printVector(result, MATRIX_SIZE);

        printf("\nElapsed Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}