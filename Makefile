CXX      := g++
INCLUDES := -Iinclude
OUTDIR   := build

# Per-stage flags
FLAGS_BASELINE := -O2 -fno-tree-vectorize
FLAGS_AVX2     := -O3 -march=native -mavx2 -mfma
FLAGS_THREADED := -O3 -march=native -mavx2 -mfma -fopenmp

.PHONY: all naive blocked avx2 aligned threaded clean

all: naive blocked avx2 aligned threaded

naive: | $(OUTDIR)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) src/kernels/naive.cpp src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/naive

blocked: | $(OUTDIR)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) src/kernels/blocked.cpp src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/blocked

avx2: | $(OUTDIR)
	$(CXX) $(FLAGS_AVX2) $(INCLUDES) src/kernels/avx2.cpp src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/avx2

aligned: | $(OUTDIR)
	$(CXX) $(FLAGS_AVX2) $(INCLUDES) src/kernels/aligned.cpp src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/aligned

threaded: | $(OUTDIR)
	$(CXX) $(FLAGS_THREADED) $(INCLUDES) src/kernels/threaded.cpp src/benchmark.cpp src/utils.cpp -o $(OUTDIR)/threaded

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)
