#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
struct Context {
    int rank;
    int size;
    int count;
    int *send_buff;
    int *recv_buff;
};

double do_allgather(Context *ctx) {
    double elapsed_time = - MPI_Wtime();
    MPI_Allgather(
        ctx->send_buff, ctx->count, MPI_INT,
        ctx->recv_buff, ctx->count, MPI_INT,
        MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_iallgather(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();

    MPI_Iallgather(
        ctx->send_buff, ctx->count, MPI_INT,
        ctx->recv_buff, ctx->count, MPI_INT,
        MPI_COMM_WORLD, &request
    );

    MPI_Wait(&request, MPI_STATUS_IGNORE);
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void cleanup(Context *ctx, int* host_buff) {
    cudaFree(ctx->send_buff);
    cudaFree(ctx->recv_buff);
    free(host_buff);
}

void check_recv_buff(Context *ctx, int* host_buff) {
    // assert that the recv_buff is the same as the send_buff
    cudaMemcpy(host_buff, ctx->recv_buff, ctx->count * ctx->size * sizeof(int), cudaMemcpyDeviceToHost);

    int i = 0;
    for (int rank = 0; rank < ctx->size; ++rank) {
        for (int k = 0; k < ctx->count; ++k) {
            if (host_buff[i] != rank) {
                fprintf(stderr, "rank: %d, recv_buff[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff[i], rank);
                fflush(stderr);
                abort();
            }
            i++;
        }
    }
}

int main(int argc, char** argv) {
    // sleep for 20 seconds
    sleep(0);

    Context ctx;
    ctx.count = 1024 * 1024 * 128;

    // initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    // Using cudaMalloc, we get the following error, so use cudaMallocHost
    //     ERROR ibv_reg_mr(addr:0x7f785ba00000 size:8192) failed: Bad address)
    //     tl_sharp_coll.c:111  TL_SHARP ERROR ucc_rcache_get failed
    cudaMallocHost((void**)&ctx.send_buff, ctx.count * sizeof(int));
    cudaMallocHost((void**)&ctx.recv_buff, ctx.count * ctx.size * sizeof(int));

    int* host_buff = (int*)malloc(ctx.count * ctx.size * sizeof(int));
    for (int i = 0; i < ctx.count; ++i) {
        host_buff[i] = ctx.rank;
    }
    cudaMemcpy(ctx.send_buff, host_buff, ctx.count * sizeof(int), cudaMemcpyHostToDevice);

    double elapsed_time;


    printf("rank: %d, start iallgather\n", ctx.rank);
    elapsed_time = do_iallgather(&ctx);
    printf("rank: %d, end iallgather, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx, host_buff);

    printf("rank: %d, start allgather\n", ctx.rank);
    elapsed_time = do_allgather(&ctx);
    printf("rank: %d, end allgather, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx, host_buff);

    sleep(30);
    fflush(stdout);
    cleanup(&ctx, host_buff);
    MPI_Finalize();

    return 0;
}
