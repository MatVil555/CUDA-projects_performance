# =====================================================================
# Makefile for the transpose optimization ladder
#
#   make                       build (set ARCH for your GPU)
#   make ARCH=sm_75            build for a Colab T4
#   make run                   build and run at the default size
#   make run M=8192 N=4096 ITERS=20
#   make banks                 CPU bank-conflict simulation -- checks the
#                              central claim of step 3 with NO GPU needed
#   make smem                  shared memory per kernel (4096 vs 4224 B)
#   make regs                  registers and spill traffic per kernel
#   make profile-step3         Nsight Compute report for one kernel
#   make metrics-step3         key metrics straight to the terminal
#   make clean
#
# ARCH must match your GPU or the binary will not run:
#   nvidia-smi --query-gpu=compute_cap --format=csv
#     7.0 -> sm_70 (V100)     7.5 -> sm_75 (T4)
#     8.0 -> sm_80 (A100)     8.6 -> sm_86 (RTX 30xx)
#     8.9 -> sm_89 (RTX 40xx) 9.0 -> sm_90 (H100)
# =====================================================================

NVCC      ?= nvcc
CXX       ?= g++
ARCH      ?= sm_80
M         ?= 4096
N         ?= 2048
ITERS     ?= 50

TARGET    := transpose_bench
BANKSIM   := bank_sim
SRC_DIR   := src
INC_DIR   := include
TOOL_DIR  := tools
BUILD_DIR := build

# -lineinfo costs nothing at run time and lets Nsight Compute attribute
# stalls and memory transactions back to individual source lines. Always
# profile with it; without it you get per-kernel totals and have to guess.
NVCC_FLAGS := -O3 -std=c++14 -arch=$(ARCH) -I$(INC_DIR) -lineinfo

# main.cu last: it is the only translation unit with host entry code, and
# listing the kernels first makes the link order obvious.
SOURCES := $(SRC_DIR)/transpose_1_naive.cu \
           $(SRC_DIR)/transpose_2_tiled.cu \
           $(SRC_DIR)/transpose_3_padded.cu \
           $(SRC_DIR)/copy_reference.cu \
           $(SRC_DIR)/main.cu

HEADERS := $(INC_DIR)/transpose_common.cuh

# Kernel sources only. 
# own, so the smem/regs targets skip it.
KERNEL_SRCS := $(SRC_DIR)/transpose_1_naive.cu \
               $(SRC_DIR)/transpose_2_tiled.cu \
               $(SRC_DIR)/transpose_3_padded.cu \
               $(SRC_DIR)/copy_reference.cu

# Coalescing is the whole story for steps 1->2, so sectors-per-request on
# the STORE is the metric to watch (expect ~32 for step 1, ~4 after).
# Bank conflicts are the story for 2->3.
KEY_METRICS := \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active

.PHONY: all run banks smem regs clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $@ $(LIBS)

run: $(TARGET)
	./$(TARGET) $(M) $(N) $(ITERS)

# ---------------------------------------------------------------------
# Inspection targets. The claims in the kernel comments are checkable, and
# these are how you check them.
# ---------------------------------------------------------------------

# The most useful target here, and the only one that needs NO GPU at all.
# Exhaustively enumerates which bank each lane touches, for row stride 32
# and 33, and asserts the documented conflict degrees. Exits non-zero if
# the arithmetic behind step 3 ever stops holding.
banks: $(BANKSIM)
	./$(BANKSIM)

$(BANKSIM): $(TOOL_DIR)/bank_sim.cpp
	$(CXX) -O2 -std=c++14 -Wall -Wextra $< -o $@

# Shared memory per kernel: the difference between step 2 and step 3 is
# exactly 128 bytes (4096 -> 4224), which is the price of removing a
# 32-way bank conflict.
smem:
	@for f in $(KERNEL_SRCS); do \
	  $(NVCC) $(NVCC_FLAGS) -Xptxas -v -c $$f -o /dev/null 2>&1 \
	    | grep -E 'entry function|smem' ; \
	done

# Registers and, more importantly, spill traffic. A non-zero spill figure
# means the kernel is secretly hitting DRAM.
regs:
	@for f in $(KERNEL_SRCS); do \
	  $(NVCC) $(NVCC_FLAGS) -Xptxas -v -c $$f -o /dev/null 2>&1 \
	    | grep -E 'entry function|registers|spill' ; \
	done

# One kernel per report: mixed reports are hard to read. One iteration is
# enough, since ncu replays and serializes kernels -- timings taken inside
# a profile are not benchmarks. Take those from `make run`.
reports:
	@mkdir -p reports

profile-step1: $(TARGET) | reports
	ncu --set full -f -k regex:transpose_l1 -o reports/step1 ./$(TARGET) $(M) $(N) 1
profile-step2: $(TARGET) | reports
	ncu --set full -f -k regex:transpose_l2 -o reports/step2 ./$(TARGET) $(M) $(N) 1
profile-step3: $(TARGET) | reports
	ncu --set full -f -k regex:transpose_l3 -o reports/step3 ./$(TARGET) $(M) $(N) 1
profile-copy:  $(TARGET) | reports
	ncu --set full -f -k regex:copy_kernel  -o reports/copy  ./$(TARGET) $(M) $(N) 1

metrics-step1: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:transpose_l1 ./$(TARGET) $(M) $(N) 1
metrics-step2: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:transpose_l2 ./$(TARGET) $(M) $(N) 1
metrics-step3: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:transpose_l3 ./$(TARGET) $(M) $(N) 1

clean:
	rm -f $(TARGET) $(BANKSIM)
	rm -rf $(BUILD_DIR) reports
