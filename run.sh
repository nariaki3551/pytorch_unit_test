if [ -z "$BIN" ]; then
    echo "BIN is not set. Please set it before running make."
    exit 1
fi
LAUNCHER=/data/fsdp_overlap/mcast/muliticast-based-allgather/build/bin/mpirun

BASE_ENV="""
-x LD_LIBRARY_PATH
-x PATH
-x TORCH_DISTRIBUTED_DEBUG=DETAIL
-x TORCH_CPP_LOG_LEVEL=INFO
-x OMP_NUM_THREADS=1
"""

UCC_ENV="""
--mca coll_ucc_enable 1
--mca coll_ucc_priority 100
-x UCC_MIN_TEAM_SIZE=2
"""

UCC_SPIN_ENV="""
-x UCC_TL_SPIN_ALLGATHER_MCAST_ROOTS=1
-x UCC_TL_SPIN_LINK_BW=12.5
-x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=$((1024 * 1024 * 1024 * 2))
"""

UCC_TUNING_ENV="""
-x UCC_TL_SPIN_TUNE=allgather:inf
-x UCC_TL_SHARP_TUNE=reduce_scatter:inf#allgather:inf#allreduce:inf
-x UCC_CL_BASIC_TLS=sharp,spin,ucp
"""
UCC_TUNING_ENV="${UCC_TUNING_ENV} -x UCC_CL_BASIC_TLS=sharp,spin,ucp"
# UCC_TUNING_ENV="${UCC_TUNING_ENV} -x UCC_CL_BASIC_TLS=ucp"

SHARP_ENV="""
-x SHARP_COLL_ENABLE_SAT=1
-x UCC_TL_SHARP_MIN_TEAM_SIZE=2
"""

VERBOSE_ENV="""
--mca coll_ucc_verbose 3
-x UCC_LOG_LEVEL=trace
-x UCC_TL_LOG_LEVEL=trace
-x SHARP_COLL_LOG_LEVEL=5
-x UCC_TL_SHARP_VERBOSE=3
"""

DEBUG_ENV="""
--mca mpi_show_mca_params all
--mca mpi_show_handle_leaks 1
--mca mpi_show_pointer_leaks 1
--mca mpi_leave_pinned 1
--mca coll_base_verbose 100
"""

ppn=1
$LAUNCHER \
  -N $ppn --host snail03:$ppn,snail02:$ppn \
  -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
  --mca btl_openib_if_include mlx5_2:1 \
  -x UCC_TL_SHARP_DEVICES=mlx5_2 \
  $BASE_ENV $VERBOSE_ENV $UCC_ENV $UCC_SPIN_ENV $UCC_TUNING_ENV $SHARP_ENV \
  $BIN $BIN_ARGS \
  | tee log
