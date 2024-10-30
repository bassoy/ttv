
ifeq ($(MACH_FLAG), TUHH)
    BLAS_DIR:=${HOME}/local
else
    BLAS_DIR:=/usr/local
endif

BLAS_INC:=-I${BLAS_DIR}/include/blis
BLAS_LIB:=${BLAS_DIR}/lib/libblis-mt.a -lm -fopenmp
#BLAS_LIB:=${BLAS_DIR}/lib/libblis.a -lm -fopenmp -lrt
BLAS_FLAGS:=-DUSE_BLIS
