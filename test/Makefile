CXX :=g++
CXX :=clang++

CXX_FLAGS +=-Wextra -Wall -Wpedantic -Ofast -std=c++17 -pthread -fopenmp -DUSE_MKLBLAS # -DUSE_OPENBLAS -DUSE_MKLBLAS
# -g -O0 -fprofile-arcs -ftest-coverage -fno-inline 

INTEL_DIR :=/opt/intel/mkl
INTEL_INC :=-DMKL_ILP64 -m64 -I$(INTEL_DIR)/include
INTEL_LIB :=-Wl,--start-group ${INTEL_DIR}/lib/intel64/libmkl_intel_ilp64.a ${INTEL_DIR}/lib/intel64/libmkl_gnu_thread.a ${INTEL_DIR}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

GTEST_LIB :=-lgtest -lpthread

TLIB_INC :=-I../include

#$(INTEL_INC)
INCS +=$(TLIB_INC) -Iinclude
LIBS +=$(GTEST_LIB) -lgomp -lpthread -lm -lopenblas

CFILES2 = $(wildcard src/*.cpp)
OBJS5   = $(notdir $(CFILES2))
OBJS6   = $(patsubst %.cpp,%.o,$(OBJS5))
OBJSC   = $(addprefix build/,$(OBJS6))


TARGET := bin/main
$(TARGET): $(OBJSC)
	$(CXX) $(CXX_FLAGS) $(OBJSC) $(LIBS) -o $(TARGET)
#	$(CXX) --version

build/%.o : src/%.cpp
	$(CXX) $(CXX_FLAGS) -c $(INCS)  $< -o $@

.PHONY : clean

clean:
	rm -f $(TARGET) build/*.* 

