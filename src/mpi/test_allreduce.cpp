#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // sleep for 20 seconds
    sleep(20);

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // sleep for 30 seconds
    // sleep(30);

    int count = 1024 * 1024 * 64;
    int* send_buff = (int*)malloc(count * sizeof(int));
    for (int i = 0; i < count; ++i) {
        send_buff[i] = rank;
    }
    int* recv_buff = (int*)malloc(count * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);
    sleep(1);
    printf("rank: %d, start allreduce\n", rank);
    for( int i = 0; i < 10; ++i ) {
        MPI_Allreduce(send_buff, recv_buff, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    printf("rank: %d, end allreduce\n", rank);
    sleep(1);

    free(send_buff);
    free(recv_buff);

    MPI_Finalize();

    return 0;
}
