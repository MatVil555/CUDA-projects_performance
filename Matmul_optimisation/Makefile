# =====================================================================
# Makefile for the matmul profiling exercises
#
# Each exercise is a standalone program built from a single .cu file, so
# you can build, run and profile any step on its own:
#
#   make matmul_3                 build one exercise
#   make                          build all of them
#   make run                      build all, run all, print the ladder
#   make run N=2048 ITERS=20      same at a different size
#   make DEBUG=1 matmul_3         build with the CUDA_CALL/CUDA_CHECK
#                                 error checking switched on
#   make profile-matmul_3         full Nsight Compute report
#   make metrics-matmul_3         just the metrics that matter, on stdout
#   make clean
#
# ARCH must match your GPU or the binary will not run:
#   nvidia-smi --query-gpu=compute_cap --format=csv
#     8.0 -> sm_80 (A100)     8.6 -> sm_86 (RTX 30xx)
#     8.9 -> sm_89 (RTX 40xx) 9.0 -> sm_90 (H100)
# =====================================================================

NVCC  ?= nvcc
ARCH  ?= sm_80
N     ?= 1024
ITERS ?= 10
DEBUG ?= 0

NVCC_FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo


ifeq ($(DEBUG),1)
NVCC_FLAGS += -DDEBUG
endif

EXERCISES := matmul_1 matmul_2 matmul_3 matmul_4 \
             matmul_5 matmul_6 matmul_7 matmul_8


KEY_METRICS := \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static

.PHONY: all run clean $(addprefix profile-,$(EXERCISES)) $(addprefix metrics-,$(EXERCISES))

all: $(EXERCISES)

# matmul_8 is the only exercise that needs cuBLAS. This explicit rule
# takes precedence over the pattern rule below.
matmul_8: matmul_8.cu matmul_common.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcublas

matmul_%: matmul_%.cu matmul_common.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Run the whole ladder in order. Exercises 3-7 require N to be a multiple
# of their tile size (32, 64 or 128) and will tell you so instead of
# producing a wrong answer -- the default N=1024 satisfies all of them.
run: all
	@for e in $(EXERCISES); do ./$$e $(N) $(ITERS) || exit 1; done

reports:
	@mkdir -p reports

# Full report, viewable with `ncu-ui reports/<name>.ncu-rep`. One
# iteration is enough: profiling replays the kernel anyway, and ncu
# serializes execution, so timings inside a profile are not benchmarks.
profile-%: % | reports
	ncu --set full -f -o reports/$* ./$* $(N) 1

# Same idea, but prints the short list of metrics straight to the
# terminal -- the fast loop when you are iterating on one kernel.
metrics-%: %
	ncu --metrics $(KEY_METRICS) ./$* $(N) 1

clean:
	rm -f $(EXERCISES)
	rm -rf reports
