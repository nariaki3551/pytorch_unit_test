#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct Context {
    int rank;
    int size;
    int count;
    int* send_buff;
    int* recv_buff;
};

double do_reduce_scatter(Context *ctx) {
    double elapsed_time = - MPI_Wtime();
    for (int i = 0; i < 1; ++i) {
        MPI_Reduce_scatter_block(
            ctx->send_buff, ctx->recv_buff, ctx->count,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD
        );
    }
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_ireduce_scatter(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();
    for (int i = 0; i < 1; ++i) {
        MPI_Ireduce_scatter_block(
            ctx->send_buff, ctx->recv_buff, ctx->count,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD, &request
        );
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void cleanup(Context *ctx) {
    free(ctx->send_buff);
    free(ctx->recv_buff);
}

void check_recv_buff(Context *ctx) {
    int sum = 0;
    for (int i = 0; i < ctx->size; ++i) {
        sum += i;
    }
    for (int i = 0; i < ctx->count; ++i) {
        if (ctx->recv_buff[i] != sum) {
            printf("rank: %d, recv_buff[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff[i], sum);
            abort();
        }
    }
}

int main(int argc, char** argv) {
    // sleep for 20 seconds
    // sleep(20);

    // user setting
    Context ctx;
    // ctx.count = 1024 * 1024 * 128;
    ctx.count = 64;

    // initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    posix_memalign((void**)&ctx.send_buff, 4096, ctx.count * ctx.size * sizeof(int));
    posix_memalign((void**)&ctx.recv_buff, 4096, ctx.count * sizeof(int));

    for (int i = 0; i < ctx.count * ctx.size; ++i) {
        ctx.send_buff[i] = ctx.rank;
    }

    double elapsed_time;

    // printf("rank: %d, start reducescatter\n", ctx.rank);
    // elapsed_time = do_reduce_scatter(&ctx);
    // printf("rank: %d, end reducescatter, time: %f\n", ctx.rank, elapsed_time);
    // check_recv_buff(&ctx);

    printf("rank: %d, start ireducescatter\n", ctx.rank);
    elapsed_time = do_ireduce_scatter(&ctx);
    printf("rank: %d, end ireducescatter, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx);

    printf("rank: %d, finish all\n", ctx.rank);
    cleanup(&ctx);
    MPI_Finalize();

    return 0;
}
