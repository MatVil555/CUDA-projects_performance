# =====================================================================
# Makefile for the reduction optimization ladder
#
#   make                       build (set ARCH for your GPU)
#   make ARCH=sm_75            build for a Colab T4
#   make run                   build and run at the default size
#   make run N=8000000 ITERS=20
#   make ptx                   dump PTX per kernel (see `make instr`)
#   make instr                 count rem/and/bar.sync/shfl per kernel --
#                              the evidence behind the claims in
#                              reduce_1_interleaved.cu
#   make regs                  per-kernel register + spill report
#   make profile-step3         Nsight Compute report for one kernel
#   make metrics-step3         key metrics straight to the terminal
#   make clean
#
# ARCH must match your GPU or the binary will not run:
#   nvidia-smi --query-gpu=compute_cap --format=csv
#     7.0 -> sm_70 (V100)     7.5 -> sm_75 (T4)
#     8.0 -> sm_80 (A100)     8.6 -> sm_86 (RTX 30xx)
#     8.9 -> sm_89 (RTX 40xx) 9.0 -> sm_90 (H100)
#
# TWO COMPILERS, ONE LINK.
# Everything lives in one flat directory and the build is split by what
# each file actually needs:
#   *.cu      -> nvcc, because they contain __global__ kernels and <<<>>>
#   main.cpp  -> the plain host compiler; it only ever calls the CUDA
#                runtime and cuBLAS through their C APIs, so it needs no
#                device compilation at all. It does need the toolkit
#                headers, hence -I$(CUDA_PATH)/include.
# The objects are then linked by nvcc, which pulls in the CUDA runtime
# and the device-code fatbinaries for us.
#
# Keeping the driver off nvcc is deliberate: it compiles faster, and it
# proves the host side of the program has no CUDA-language dependency --
# only an API one.
# =====================================================================

NVCC      ?= nvcc
CXX       ?= g++
ARCH      ?= sm_80
N         ?= 32768000
ITERS     ?= 50

# main.cpp needs the toolkit headers even though it never invokes nvcc, so
# locate the install root from wherever nvcc lives: <prefix>/bin/nvcc ->
# <prefix>. Falls back to the conventional path if nvcc is not on PATH.
# Override explicitly if your toolkit is somewhere unusual:
#   make CUDA_PATH=/opt/cuda-12.4
NVCC_BIN := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(NVCC_BIN),)
  CUDA_PATH ?= /usr/local/cuda
else
  CUDA_PATH ?= $(abspath $(dir $(NVCC_BIN))..)
endif

TARGET    := reduction_bench
BUILD_DIR := build

# -lineinfo costs nothing at run time and lets Nsight Compute attribute
# stalls and memory transactions back to individual source lines. Always
# profile with it; without it you get per-kernel totals and have to guess.
NVCC_FLAGS := -O3 -std=c++14 -arch=$(ARCH) -I. -lineinfo
CXXFLAGS   := -O3 -std=c++14 -I. -I$(CUDA_PATH)/include
LIBS       := -lcublas

# Device translation units: one per rung of the ladder, plus the shared
# stage-2 kernel.
CU_SOURCES  := reduce_1_interleaved.cu \
               reduce_2_sequential.cu \
               reduce_3_shuffle.cu \
               reduce_4_cublas.cu \
               reduce_final.cu

# Host translation unit: the driver, and nothing else.
CXX_SOURCES := main.cpp

HEADERS := reduction_common.h

OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(CU_SOURCES)) \
           $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CXX_SOURCES))

# Kernel sources only -- main.cpp and the cuBLAS step define no __global__
# worth inspecting, so the PTX/register targets skip them.
KERNEL_SRCS := reduce_1_interleaved.cu \
               reduce_2_sequential.cu \
               reduce_3_shuffle.cu \
               reduce_final.cu

KEY_METRICS := \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
dram__bytes_read.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread

.PHONY: all run clean ptx instr regs profile-% metrics-%

all: $(TARGET)

# nvcc does the link so the device fatbinaries and the CUDA runtime are
# resolved without naming them by hand.
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $@ $(LIBS)

$(BUILD_DIR)/%.cu.o: %.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(N) $(ITERS)

# ---------------------------------------------------------------------
# Inspection targets. The performance claims in the kernel comments are
# checkable, and these are how you check them.
# ---------------------------------------------------------------------
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Per-kernel PTX. Useful on its own; `make instr` summarizes it.
ptx: | $(BUILD_DIR)
	@for f in $(KERNEL_SRCS); do \
	  b=$$(basename $$f .cu); \
	  $(NVCC) -O3 -arch=$(ARCH) -I. -ptx $$f -o $(BUILD_DIR)/$$b.ptx; \
	  echo "$(BUILD_DIR)/$$b.ptx"; \
	done

# Instruction counts behind the divergence / modulo discussion in
# reduce_1_interleaved.cu. Expect step 1: 0 rem, 8 and.b32, 9 bar.sync;
# step 2: 0 rem, 0 and.b32, 9 bar.sync; step 3: 1 bar.sync, 10 shfl.
instr: ptx
	@echo ""
	@echo "kernel                          rem  and.b32  bar.sync  shfl"
	@echo "-------------------------------------------------------------"
	@for f in $(KERNEL_SRCS); do \
	  b=$$(basename $$f .cu); p=$(BUILD_DIR)/$$b.ptx; \
	  printf "%-30s %4s %8s %9s %5s\n" $$b \
	    $$(grep -cE 'rem\.[us]32' $$p || true) \
	    $$(grep -c 'and\.b32' $$p || true) \
	    $$(grep -c 'bar\.sync' $$p || true) \
	    $$(grep -c 'shfl\.sync' $$p || true); \
	done
	@echo ""

# Register use and, more importantly, spill traffic. A non-zero spill
# figure means the kernel is secretly hitting DRAM.
regs:
	@$(NVCC) $(NVCC_FLAGS) -Xptxas -v -c reduce_3_shuffle.cu \
	    -o /dev/null 2>&1 | grep -E 'entry|registers|spill'
	@$(NVCC) $(NVCC_FLAGS) -Xptxas -v -c reduce_1_interleaved.cu \
	    -o /dev/null 2>&1 | grep -E 'entry|registers|spill'

# One kernel per report: mixed reports are hard to read. One iteration is
# enough, since ncu replays and serializes kernels -- timings taken inside
# a profile are not benchmarks. Take those from `make run`.
reports:
	@mkdir -p reports

profile-step1: $(TARGET) | reports
	ncu --set full -f -k regex:reduce_l1 -o reports/step1 ./$(TARGET) $(N) 1
profile-step2: $(TARGET) | reports
	ncu --set full -f -k regex:reduce_l2 -o reports/step2 ./$(TARGET) $(N) 1
profile-step3: $(TARGET) | reports
	ncu --set full -f -k regex:reduce_l3 -o reports/step3 ./$(TARGET) $(N) 1
profile-step4: $(TARGET) | reports
	ncu --set full -f -k regex:"asum|dot|nrm" -o reports/step4 ./$(TARGET) $(N) 1

metrics-step1: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:reduce_l1 ./$(TARGET) $(N) 1
metrics-step2: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:reduce_l2 ./$(TARGET) $(N) 1
metrics-step3: $(TARGET)
	ncu --metrics $(KEY_METRICS) -k regex:reduce_l3 ./$(TARGET) $(N) 1

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR) reports
