#GTEST_DIR :=$(HOME)/wt/github/googletest
#GTEST_INC :=-I$(GTEST_DIR)/googletest/include
#GTEST_LIB :=-L$(GTEST_DIR)/build/lib -lgtest -lpthread
GTEST_DIR := ${HOME}/local
GTEST_INC := -I${GTEST_DIR}/include
GTEST_LIB := -L${GTEST_DIR}/lib64 -lgtest# -lpthread
