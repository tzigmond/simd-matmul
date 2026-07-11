CXX      := g++
INCLUDES := -Iinclude
OUTDIR   := build

# Per-stage flags
FLAGS_BASELINE := -O2 -fno-tree-vectorize
FLAGS_AVX2     := -O3 -march=native -mavx2 -mfma
FLAGS_THREADED := -O3 -march=native -mavx2 -mfma -fopenmp

.PHONY: all naive blocked avx2 aligned threaded clean

all: naive blocked avx2 aligned threaded

KERNELS := src/kernels/naive.cpp src/kernels/blocked.cpp src/kernels/avx2.cpp \
           src/kernels/aligned.cpp src/kernels/threaded.cpp

# Each binary links all kernels so benchmark.cpp can dispatch any of them.
# The compile flags differ per target — that's where the optimization story lives.

naive: | $(OUTDIR)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) $(KERNELS) src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/naive

blocked: | $(OUTDIR)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) $(KERNELS) src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/blocked

avx2: | $(OUTDIR)
	$(CXX) $(FLAGS_AVX2) $(INCLUDES) $(KERNELS) src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/avx2

aligned: | $(OUTDIR)
	$(CXX) $(FLAGS_AVX2) $(INCLUDES) $(KERNELS) src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/aligned

threaded: | $(OUTDIR)
	$(CXX) $(FLAGS_THREADED) $(INCLUDES) $(KERNELS) src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/threaded

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)
