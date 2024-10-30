#BLAS_INC:=-I/usr/include/x86_64-linux-gnu/openblas64-pthread
#BLAS_LIB:=-lopenblas
BLAS_DIR:=${HOME}/local
BLAS_INC:=-I${BLAS_DIR}/include
BLAS_LIB:=-L${BLAS_DIR}/lib -lopenblas
BLAS_FLAGS:=-DUSE_OPENBLAS
