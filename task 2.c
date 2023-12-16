#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 4

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void cannonMatrixMultiply(int *A, int *B, int *C, int block_size, int size, int rank) {
    int *buffer_A = (int *)malloc(block_size * sizeof(int));
    int *buffer_B = (int *)malloc(block_size * sizeof(int));

    MPI_Status status;

    // Shift A and B in a circular fashion
    int source_A = (rank / size + rank) % size;
    int dest_A = (rank - rank / size + size) % size;
    int source_B = (rank - rank / size + size) % size;
    int dest_B = (rank / size + rank) % size;

    // Initial scatter of A and B to all processes
    MPI_Scatter(A, block_size, MPI_INT, buffer_A, block_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, block_size, MPI_INT, buffer_B, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Main Cannon algorithm
    for (int k = 0; k < size; k++) {
        // Local matrix multiplication
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                C[i * block_size + j] += buffer_A[i] * buffer_B[j];
            }
        }

        // Shift A and B in a circular fashion
        MPI_Sendrecv_replace(buffer_A, block_size, MPI_INT, dest_A, 0, source_A, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv_replace(buffer_B, block_size, MPI_INT, dest_B, 0, source_B, 0, MPI_COMM_WORLD, &status);
    }

    free(buffer_A);
    free(buffer_B);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = MATRIX_SIZE / size;
    int A[MATRIX_SIZE * MATRIX_SIZE] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    int B[MATRIX_SIZE * MATRIX_SIZE] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    int C[MATRIX_SIZE * MATRIX_SIZE] = {0}; // Initialize C with zeros

    double start_time, end_time;

    if (rank == 0) {
        printf("Matrix A:\n");
        printMatrix(A, MATRIX_SIZE, MATRIX_SIZE);

        printf("\nMatrix B:\n");
        printMatrix(B, MATRIX_SIZE, MATRIX_SIZE);
    }

    start_time = MPI_Wtime(); // Measure start time

    // Broadcast matrices A and B to all processes
    MPI_Bcast(A, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication using Cannon algorithm
    cannonMatrixMultiply(A, B, C, block_size, size, rank);

    end_time = MPI_Wtime(); // Measure end time

    if (rank == 0) {
        printf("\nResult Matrix C:\n");
        printMatrix(C, MATRIX_SIZE, MATRIX_SIZE);

        printf("\nElapsed Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
