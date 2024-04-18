CXX :=g++
# CXX :=clang++

CXX_FLAGS +=-Wextra -Wall -Wpedantic -Ofast -std=c++17 -DUSE_MKLBLAS -fopenmp -DNDEBUG#  -DUSE_MKLBLAS -pthread  -DUSE_OPENBLAS
# -g -O0 -fprofile-arcs -ftest-coverage -fno-inline 


OPENBLAS_INC := -I/usr/include/x86_64-linux-gnu/openblas64-pthread
OPENBLAS_LIB := -lopenblas -lpthread -lm

#INTEL_DIR :=/opt/intel/mkl
INTEL_LIB_DIR := /usr/lib/x86_64-linux-gnu
INTEL_INC :=-DMKL_ILP64 -m64 #-I$(INTEL_DIR)/include
INTEL_LIB :=-Wl,--start-group ${INTEL_LIB_DIR}/libmkl_intel_ilp64.a ${INTEL_LIB_DIR}/libmkl_gnu_thread.a ${INTEL_LIB_DIR}/libmkl_core.a -Wl,--end-group -lm -ldl -m64 # /lib/intel64 -lgomp -lpthread -lm -ldl

GTEST_LIB :=-lgtest -lpthread

TLIB_INC :=-I../include

#$(INTEL_INC)
INCS +=$(TLIB_INC) $(INTEL_INC) -Iinclude # $(OPENBLAS_INC) 
LIBS +=$(GTEST_LIB) $(INTEL_LIB) -lgomp -lpthread -lm # $(OPENBLAS_LIB)

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

