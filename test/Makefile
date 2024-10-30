CXX :=g++

# make -j BLAS_FLAG=BLIS MACH_FLAG=TUHH 

CXX_FLAGS +=-Wextra -Wall -Wpedantic -O3 -std=c++17 -fopenmp

ifeq ($(BLAS_FLAG), OPENBLAS)
  include ../openblas.mk
else ifeq ($(BLAS_FLAG), MKL)
  include ../mkl.mk
else ifeq ($(BLAS_FLAG), BLIS)
  include ../blis.mk
endif

include gtest.mk

TLIB_INC :=-I../include -Iinclude

CXX_FLAGS += $(BLAS_FLAGS)
INCS      += $(GTEST_INC) $(TLIB_INC) $(BLAS_INC)
LIBS      += $(GTEST_LIB) $(BLAS_LIB)

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

