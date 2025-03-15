#include <mpi.h>
#include <stdio.h>
#include <mpi-ext.h> 

int main(int argc, char **argv) {
    int provided, cuda_supported;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

#if defined(MPIX_CUDA_AWARE_SUPPORT)
    printf("MPIX_CUDA_AWARE_SUPPORT is defined.\n");
    cuda_supported = MPIX_Query_cuda_support();
#else
    printf("MPIX_CUDA_AWARE_SUPPORT is not defined.\n");
    cuda_supported = 0;
#endif

    if (cuda_supported) {
        printf("MPI CUDA support is ENABLED.\n");
    } else {
        printf("MPI CUDA support is DISABLED.\n");
    }

    MPI_Finalize();
    return 0;
}
