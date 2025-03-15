#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>

struct Context {
    int rank;
    int size;
    int count;
    int *send_buff_ag;
    int *recv_buff_ag;
    int *send_buff_rs;
    int *recv_buff_rs;
    int niter;
};

double do_allgather(Context *ctx) {
    double elapsed_time = -MPI_Wtime();
    MPI_Allgather(
        ctx->send_buff_ag, ctx->count, MPI_INT,
        ctx->recv_buff_ag, ctx->count, MPI_INT,
        MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_iallgather(Context *ctx, MPI_Request *request) {
    double elapsed_time = -MPI_Wtime();
    MPI_Iallgather(
        ctx->send_buff_ag, ctx->count, MPI_INT,
        ctx->recv_buff_ag, ctx->count, MPI_INT,
        MPI_COMM_WORLD, request
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_reduce_scatter(Context *ctx) {
    double elapsed_time = -MPI_Wtime();

    for (int i = 0; i < 1; ++i) {
        MPI_Reduce_scatter_block(
            ctx->send_buff_rs, ctx->recv_buff_rs, ctx->count,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD
        );
    }
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_ireduce_scatter(Context *ctx, MPI_Request *request) {
    double elapsed_time = -MPI_Wtime();

    for (int i = 0; i < 1; ++i) {
        MPI_Ireduce_scatter_block(
            ctx->send_buff_rs, ctx->recv_buff_rs, ctx->count,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD, request
        );
    }
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_sequential(Context *ctx) {
    double elapsed_time = -MPI_Wtime();

    MPI_Request request_ag, request_rs;

    do_iallgather(ctx, &request_ag);
    MPI_Wait(&request_ag, MPI_STATUS_IGNORE);

    do_ireduce_scatter(ctx, &request_rs);
    MPI_Wait(&request_rs, MPI_STATUS_IGNORE);

    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_overlap(Context *ctx) {
    MPI_Request request_ag, request_rs;

    double elapsed_time = -MPI_Wtime();

    do_iallgather(ctx, &request_ag);
    do_ireduce_scatter(ctx, &request_rs);

    MPI_Wait(&request_ag, MPI_STATUS_IGNORE);
    MPI_Wait(&request_rs, MPI_STATUS_IGNORE);

    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void cleanup(Context *ctx, int* host_buff_ag, int* host_buff_rs) {
    cudaFree(ctx->send_buff_ag);
    cudaFree(ctx->recv_buff_ag);
    cudaFree(ctx->send_buff_rs);
    cudaFree(ctx->recv_buff_rs);
    free(host_buff_ag);
    free(host_buff_rs);
}

void check_recv_buff_ag(Context *ctx, int* host_buff) {
    // assert that the recv_buff is the same as the send_buff
    cudaMemcpy(host_buff, ctx->recv_buff_ag, ctx->count * ctx->size * sizeof(int), cudaMemcpyDeviceToHost);
    int i = 0;
    for (int rank = 0; rank < ctx->size; ++rank) {
        for (int k = 0; k < ctx->count; ++k) {
            if (ctx->recv_buff_ag[i] != rank) {
                fprintf(stderr, "rank: %d, recv_buff_ag[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff_ag[i], rank);
                fflush(stderr);
                abort();
            }
            i++;
        }
    }
}

void check_recv_buff_rs(Context *ctx, int* host_buff) {
    cudaMemcpy(host_buff, ctx->recv_buff_rs, ctx->count * sizeof(int), cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int i = 0; i < ctx->size; ++i) {
        sum += i;
    }
    for (int i = 0; i < ctx->count; ++i) {
        if (ctx->recv_buff_rs[i] != sum) {
            printf("rank: %d, recv_buff_rs[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff_rs[i], sum);
            abort();
        }
    }
}

void reset_recv_buff(Context *ctx, int* host_buff_ag, int* host_buff_rs) {
    for (int i = 0; i < ctx->count * ctx->size; ++i) {
        host_buff_ag[i] = 0;
    }
    for (int i = 0; i < ctx->count; ++i) {
        host_buff_rs[i] = 0;
    }
    cudaMemcpy(ctx->recv_buff_ag, host_buff_ag, ctx->count * ctx->size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->recv_buff_rs, host_buff_rs, ctx->count * sizeof(int), cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) {
    sleep(0);
    Context ctx;

    // user setting
    ctx.count = 1024 * 1024 * 128;
    // ctx.count = 1024 * 8;
    ctx.niter = 1;

    // initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    cudaMallocHost((void**)&ctx.send_buff_ag, ctx.count * sizeof(int));
    cudaMallocHost((void**)&ctx.recv_buff_ag, ctx.count * ctx.size * sizeof(int));
    cudaMallocHost((void**)&ctx.send_buff_rs, ctx.count * ctx.size * sizeof(int));
    cudaMallocHost((void**)&ctx.recv_buff_rs, ctx.count * sizeof(int));

    int* host_buff_ag = (int*)malloc(ctx.count * ctx.size * sizeof(int));
    int* host_buff_rs = (int*)malloc(ctx.count * ctx.size * sizeof(int));

    for (int i = 0; i < ctx.count; ++i) {
        host_buff_ag[i] = ctx.rank;
    }
    for (int i = 0; i < ctx.count * ctx.size; ++i) {
        host_buff_rs[i] = ctx.rank;
    }
    cudaMemcpy(ctx.send_buff_ag, host_buff_ag, ctx.count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.send_buff_rs, host_buff_rs, ctx.count * ctx.size * sizeof(int), cudaMemcpyHostToDevice);
    reset_recv_buff(&ctx, host_buff_ag, host_buff_rs);

    double elapsed_time;

    // warmup
    for (int i = 0; i < 2; ++i) {
        printf("rank: %d, warmup %d\n", ctx.rank, i);
        // do_sequential(&ctx);
        do_overlap(&ctx);
    }

    for( int i = 0; i < 10; ++i ) {
        // sequential
        printf("rank: %d, start sequential\n", ctx.rank);
        elapsed_time = do_sequential(&ctx);
        printf("rank: %d, end sequential, time: %f\n", ctx.rank, elapsed_time);
        check_recv_buff_ag(&ctx, host_buff_ag);
        check_recv_buff_rs(&ctx, host_buff_rs);
        reset_recv_buff(&ctx, host_buff_ag, host_buff_rs);

        // overlap
        printf("rank: %d, start overlap\n", ctx.rank);
        elapsed_time = do_overlap(&ctx);
        printf("rank: %d, end overlap, time: %f\n", ctx.rank, elapsed_time);
        check_recv_buff_ag(&ctx, host_buff_ag);
        check_recv_buff_rs(&ctx, host_buff_rs);
        reset_recv_buff(&ctx, host_buff_ag, host_buff_rs);
    }

    // finalization
    sleep(30);
    cleanup(&ctx, host_buff_ag, host_buff_rs);
    fflush(stdout);
    MPI_Finalize();

    return 0;
}
