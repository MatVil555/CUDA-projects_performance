# Build: make            -> ./softmax
# Run:   ./softmax [M] [N] [iterations]
#
# nvcc compiles main.cpp as host-only C++ (it contains no <<<>>> syntax)
# and the three .cu files as device+host code, then links everything
# together in one invocation.

NVCC     := nvcc
NVCCFLAGS:= -O3
TARGET   := softmax

SOURCES  := main.cpp softmax_l1_naive.cu softmax_l2_shared.cu softmax_l3_warp_online.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) softmax_kernels.h
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
