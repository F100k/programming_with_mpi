#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int read_matrix_flat(const char *filename, float **matrix, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    fscanf(fp, "%d %d", rows, cols);
    *matrix = (float *)malloc((*rows) * (*cols) * sizeof(float));

    for (int i = 0; i < (*rows) * (*cols); i++)
        fscanf(fp, "%f", &((*matrix)[i]));

    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0, cols = 0;
    float *A = NULL, *B = NULL, *C = NULL, *Result = NULL;

    double start = MPI_Wtime();

    if (rank == 0) {
        if (read_matrix_flat("/home/jatup/Desktop/HPC/LAB_lecture2_2025/matAsmall.txt", &A, &rows, &cols) != 0 ||
            read_matrix_flat("/home/jatup/Desktop/HPC/LAB_lecture2_2025/matBsmall.txt", &B, &rows, &cols) != 0 ||
            read_matrix_flat("/home/jatup/Desktop/HPC/LAB_lecture2_2025/solutionsmall.txt", &Result, &rows, &cols) != 0) {
            printf("Error reading matrix files\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base = rows / size;
    int rem  = rows % size;

    int local_rows = base + (rank < rem ? 1 : 0);
    int local_size = local_rows * cols;

    float *A_local = malloc(local_size * sizeof(float));
    float *B_local = malloc(local_size * sizeof(float));
    float *C_local = malloc(local_size * sizeof(float));

    if (rank == 0) {
        int offset = local_rows;

        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0);
            int count  = rows_p * cols;

            MPI_Send(A + offset * cols, count, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            MPI_Send(B + offset * cols, count, MPI_FLOAT, p, 1, MPI_COMM_WORLD);

            offset += rows_p;
        }

        for (int i = 0; i < local_size; i++) {
            A_local[i] = A[i];
            B_local[i] = B[i];
        }
    } else {
        MPI_Recv(A_local, local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_local, local_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < local_size; i++)
        C_local[i] = A_local[i] + B_local[i];

    if (rank == 0) {
        C = malloc(rows * cols * sizeof(float));

        for (int i = 0; i < local_size; i++)
            C[i] = C_local[i];

        int offset = local_rows;

        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0);
            int count  = rows_p * cols;

            MPI_Recv(C + offset * cols, count,
                     MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            offset += rows_p;
        }

        int mismatch = 0;
        for (int i = 0; i < rows * cols; i++)
            if (fabsf(C[i] - Result[i]) > 1e-5f)
                mismatch++;

        if (mismatch == 0)
            printf("Blocking MPI Done\n");
        else
            printf("Blocking MPI mismatches = %d\n", mismatch);
    } else {
        MPI_Send(C_local, local_size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    if (rank == 0)
        printf("Blocking time: %f sec\n", end - start);

    free(A_local);
    free(B_local);
    free(C_local);
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
        free(Result);
    }

    MPI_Finalize();
    return 0;
}
