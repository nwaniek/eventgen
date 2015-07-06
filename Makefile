include config.mk

CC	  	= clang
CXX	  	= clang++
LD	  	= ld
NVCC	  	= nvcc
CXX_STD	  	= c++11
WARNINGS  	= -Wall -Werror -pedantic
CFLAGS	  	= -fPIC -O3 -ffast-math
LDFLAGS	  	= ${LIBS}
OBJ_FILES   	= $(SRC:.cpp=.o)
CU_OBJ_FILES 	= $(CU_SRC:.cu=.o)


SRC		= main.cpp \
		  io.cpp

CU_SRC 		= kernels/dvs.cu \
		  kernels/frame.cu \
		  kernels/misc.cu

INCLUDES  	= -I. \
		  -Ikernels \
		  -Icontrib \
		  -I${CUDA_INCLUDE_PATH}

LIBS	  	= -L${CUDA_LIB_PATH} \
		  -lpthread -lm \
		  -lboost_system -lboost_program_options \
		  -lcuda -lcudart 


all: eventgen

eventgen: $(OBJ_FILES) ${CU_OBJ_FILES}
	${CXX} ${LDFLAGS} $^ -o $@

io.o: io.cpp io.hpp
	${CXX} ${CFLAGS} ${WARNINGS} ${INCLUDES} -std=${CXX_STD} -c io.cpp

%.o: %.cpp
	${CXX} ${CFLAGS} ${WARNINGS} ${INCLUDES} -std=${CXX_STD} -c $<

kernels/%.o: kernels/%.cu
	${NVCC} ${INCLUDES} -std=${CXX_STD} -c $< -o $@

clean:
	rm -rf main.o
	rm -rf kernels/dvs.o
