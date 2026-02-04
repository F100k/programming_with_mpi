#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int read_matrix_flat(const char *filename, float **matrix, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return -1;
    }
    *matrix = malloc((*rows) * (*cols) * sizeof(float));
    for (int i = 0; i < (*rows) * (*cols); i++)
        if (fscanf(fp, "%f", &((*matrix)[i])) != 1) break;
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
        read_matrix_flat("matAlarge.txt", &A, &rows, &cols);
        read_matrix_flat("matBlarge.txt", &B, &rows, &cols);
        read_matrix_flat("solutionAdd.txt", &Result, &rows, &cols);

        MPI_Request *dim_reqs = malloc(2 * (size - 1) * sizeof(MPI_Request));
        for (int p = 1; p < size; p++) {
            MPI_Isend(&rows, 1, MPI_INT, p, 10, MPI_COMM_WORLD, &dim_reqs[(p - 1) * 2]);
            MPI_Isend(&cols, 1, MPI_INT, p, 11, MPI_COMM_WORLD, &dim_reqs[(p - 1) * 2 + 1]);
        }
        MPI_Waitall(2 * (size - 1), dim_reqs, MPI_STATUSES_IGNORE);
        free(dim_reqs);
    } else {
        MPI_Request reqs[2];
        MPI_Irecv(&rows, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&cols, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    int base = rows / size;
    int rem = rows % size;
    int local_rows = base + (rank < rem ? 1 : 0);
    int local_size = local_rows * cols;

    float *A_local = malloc(local_size * sizeof(float));
    float *B_local = malloc(local_size * sizeof(float));
    float *C_local = malloc(local_size * sizeof(float));

    if (rank == 0) {
        MPI_Request *data_reqs = malloc(2 * (size - 1) * sizeof(MPI_Request));
        int offset = local_rows;
        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0);
            int count = rows_p * cols;
            MPI_Isend(A + offset * cols, count, MPI_FLOAT, p, 20, MPI_COMM_WORLD, &data_reqs[(p - 1) * 2]);
            MPI_Isend(B + offset * cols, count, MPI_FLOAT, p, 21, MPI_COMM_WORLD, &data_reqs[(p - 1) * 2 + 1]);
            offset += rows_p;
        }
        for (int i = 0; i < local_size; i++) {
            A_local[i] = A[i];
            B_local[i] = B[i];
        }
        MPI_Waitall(2 * (size - 1), data_reqs, MPI_STATUSES_IGNORE);
        free(data_reqs);
    } else {
        MPI_Request reqs[2];
        MPI_Irecv(A_local, local_size, MPI_FLOAT, 0, 20, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(B_local, local_size, MPI_FLOAT, 0, 21, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < local_size; i++)
        C_local[i] = A_local[i] + B_local[i];

    if (rank == 0) {
        C = malloc(rows * cols * sizeof(float));
        MPI_Request *gather_reqs = malloc((size - 1) * sizeof(MPI_Request));
        int offset = local_rows;

        for (int i = 0; i < local_size; i++) C[i] = C_local[i];

        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0);
            int count = rows_p * cols;
            MPI_Irecv(C + offset * cols, count, MPI_FLOAT, p, 30, MPI_COMM_WORLD, &gather_reqs[p - 1]);
            offset += rows_p;
        }
        MPI_Waitall(size - 1, gather_reqs, MPI_STATUSES_IGNORE);
        free(gather_reqs);

        double end = MPI_Wtime();
        printf("Non-blocking time: %f sec\n", end - start);

        FILE *out = fopen("result.txt", "w");
        if (out) {
            fprintf(out, "%d %d\n", rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                fprintf(out, "%f ", C[i]);
                if ((i + 1) % cols == 0) fprintf(out, "\n");
            }
            fclose(out);
        }
        printf("Non-blocking MPI Done and result.txt saved\n");

        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        if (Result) free(Result);
    } else {
        MPI_Request req;
        MPI_Isend(C_local, local_size, MPI_FLOAT, 0, 30, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    free(A_local); free(B_local); free(C_local);
    MPI_Finalize();
    return 0;
}