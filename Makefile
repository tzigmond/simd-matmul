CXX      := g++
INCLUDES := -Iinclude
OUTDIR   := build
OBJ      := $(OUTDIR)/obj
BIN      := $(OUTDIR)/benchmark
STD      := -std=c++17

# Each kernel object is compiled with the flags that define its stage, then all
# objects are linked into one binary. This is the whole point of the layout:
# naive/blocked are honest scalar baselines (vectorization disabled), the SIMD
# stages get AVX2+FMA, and *only* threaded.o is compiled with -fopenmp so the
# omp pragma in the shared core activates for that translation unit alone.
FLAGS_BASELINE := -O2 -march=native -fno-tree-vectorize $(STD)
FLAGS_SIMD     := -O3 -march=native -mavx2 -mfma $(STD)
FLAGS_THREADED := -O3 -march=native -mavx2 -mfma -fopenmp $(STD)
FLAGS_PLAIN    := -O2 -march=native $(STD)

OBJS := $(OBJ)/naive.o $(OBJ)/blocked.o $(OBJ)/avx2.o $(OBJ)/aligned.o \
        $(OBJ)/threaded.o $(OBJ)/openblas.o $(OBJ)/benchmark.o $(OBJ)/utils.o

.PHONY: all clean
all: $(BIN)

$(OBJ):
	mkdir -p $(OBJ)

$(OBJ)/naive.o: src/kernels/naive.cpp | $(OBJ)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) -c $< -o $@
$(OBJ)/blocked.o: src/kernels/blocked.cpp | $(OBJ)
	$(CXX) $(FLAGS_BASELINE) $(INCLUDES) -c $< -o $@
$(OBJ)/avx2.o: src/kernels/avx2.cpp | $(OBJ)
	$(CXX) $(FLAGS_SIMD) $(INCLUDES) -c $< -o $@
$(OBJ)/aligned.o: src/kernels/aligned.cpp | $(OBJ)
	$(CXX) $(FLAGS_SIMD) $(INCLUDES) -c $< -o $@
$(OBJ)/threaded.o: src/kernels/threaded.cpp | $(OBJ)
	$(CXX) $(FLAGS_THREADED) $(INCLUDES) -c $< -o $@
$(OBJ)/openblas.o: src/kernels/openblas.cpp | $(OBJ)
	$(CXX) $(FLAGS_PLAIN) $(INCLUDES) -c $< -o $@
$(OBJ)/benchmark.o: src/benchmark.cpp | $(OBJ)
	$(CXX) $(FLAGS_PLAIN) $(INCLUDES) -c $< -o $@
$(OBJ)/utils.o: src/utils.cpp | $(OBJ)
	$(CXX) $(FLAGS_PLAIN) $(INCLUDES) -c $< -o $@

# Link with the OpenMP flag so threaded.o's libgomp references resolve, and
# against OpenBLAS for the reference kernel.
$(BIN): $(OBJS)
	$(CXX) $(FLAGS_THREADED) $(OBJS) -o $(BIN) -lopenblas

clean:
	rm -rf $(OUTDIR)
