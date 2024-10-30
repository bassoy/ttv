CXX :=g++

CXX_FLAGS:=-Wall -O3 -std=c++17 -fopenmp

ifeq ($(BLAS_FLAG), OPENBLAS)
  include ../openblas.mk
else ifeq ($(BLAS_FLAG), MKL)
  include ../mkl.mk
else ifeq ($(BLAS_FLAG), BLIS)
  include ../blis.mk
endif

TLIB_INC :=-I../include

CXX_FLAGS += $(BLAS_FLAGS)
INCS += $(TLIB_INC) $(BLAS_INC)
LIBS += $(BLAS_LIB)

# Source files
SRC1 := interface1.cpp
SRC2 := interface2.cpp
SRC3 := interface3.cpp

# Object files
OBJ1 := build/$(SRC1:.cpp=.o)
OBJ2 := build/$(SRC2:.cpp=.o)
OBJ3 := build/$(SRC3:.cpp=.o)

EXE1   = bin/$(basename $(SRC1))
EXE2   = bin/$(basename $(SRC2))
EXE3   = bin/$(basename $(SRC3))

# Targets
all: $(EXE1) $(EXE2) $(EXE3)

$(EXE1): $(OBJ1)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)

$(EXE2): $(OBJ2)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)

$(EXE3): $(OBJ3)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS) 

# Generic rule to build object files
build/%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCS) -c $< -o $@

# Clean rule
clean:
	rm -f $(EXE1) $(EXE2) $(EXE3) $(OBJ1) $(OBJ2) $(OBJ3)

