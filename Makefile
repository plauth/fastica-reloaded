CXX := clang++
CXXFLAGS := -std=c++11 -O3
INC_DIRS := -I/usr/local/include/
LIB_DIRS := -L/usr/local/lib/
LDFLAGS := -lboost_program_options -lsndfile

ifeq ($(shell uname),Darwin)
	LDFLAGS_OCL := -framework OpenCL
else
	LDFLAGS_OCL := -lOpenCL
endif

all: fastica-reloaded-cpu
	
clean:
	rm -f fastica-reloaded-cpu
	
cpu:
	$(CXX) main.cpp $(CXXFLAGS) $(INC_DIRS) $(LIB_DIRS) $(LDFLAGS) -o fastica-reloaded-$@

gpu:
	$(CXX) main_gpu.cpp clmatrix.cpp clreduceaddkernel.cpp clcenterkernel.cpp $(CXXFLAGS) $(INC_DIRS) $(LIB_DIRS) $(LDFLAGS) $(LDFLAGS_OCL) -o fastica-reloaded-$@ -g

