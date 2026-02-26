#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int read_matrix_flat(const char *filename, float **matrix, int *rows, int *cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return -1;
    }
    *matrix = malloc((*rows) * (*cols) * sizeof(float));
    for (int i = 0; i < (*rows) * (*cols); i++) {
        if (fscanf(fp, "%f", &((*matrix)[i])) != 1) break;
    }
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0, cols = 0;
    float *A = NULL, *B = NULL, *C = NULL;

    double start = MPI_Wtime();

    if (rank == 0) {
        if (read_matrix_flat("matAlarge.txt", &A, &rows, &cols) != 0 ||
            read_matrix_flat("matBlarge.txt", &B, &rows, &cols) != 0) {
            printf("Error reading input files!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C = malloc(rows * cols * sizeof(float));
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int local_rows = (rows / size) + (i < (rows % size) ? 1 : 0);
        sendcounts[i] = local_rows * cols;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int my_count = sendcounts[rank];
    float *A_local = malloc(my_count * sizeof(float));
    float *B_local = malloc(my_count * sizeof(float));
    float *C_local = malloc(my_count * sizeof(float));

    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, A_local, my_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, MPI_FLOAT, B_local, my_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < my_count; i++) {
        C_local[i] = A_local[i] + B_local[i];
    }

    MPI_Gatherv(C_local, my_count, MPI_FLOAT, C, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end = MPI_Wtime();
        FILE *out = fopen("result.txt", "w");
        if (out) {
            fprintf(out, "%d %d\n", rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                fprintf(out, "%f ", C[i]);
                if ((i + 1) % cols == 0) fprintf(out, "\n");
            }
            fclose(out);
        }

        printf("Collective communication time: %f sec\n", end - start);
        printf("Collective Done and result.txt saved\n");

        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
    }

    free(A_local);
    free(B_local);
    free(C_local);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}