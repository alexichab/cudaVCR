CXX = g++
NVCC = /usr/local/cuda/bin/nvcc

CXXFLAGS = -g -O2 -msse2 -msse3 -msse4 -mmmx -I/usr/local/cuda/include
#CXXFLAGS = -g -msse2 -msse3 -msse4 -mmmx -maxrregcount=64 
NVCCFLAGS =  -std=c++17 -ccbin /usr/bin/g++ -g -G -O2 -arch=sm_75 --ptxas-options=-v

CUDA_LIBS = -lcudart -L/usr/local/cuda/lib64
CUDA_INC = -I/usr/local/cuda/include

all: mc_growth.exe

mc_growth.exe :  mc_growth.o geometry.o deform.o deform2.o deform3.o jump.o cycle.o output.o cuda_kernels.o
#	g++ $(CXXFLAGS) -Wall -lm -o mc_growth.exe  mc_growth.o geometry.o deform.o deform2.o deform3.o jump.o cycle.o output.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LIBS)

# Правило для CUDA файлов
cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c $< -o $@

# Правила для C++ файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean :
	rm -f mc_growth.exe *.o
	rm -f A[0-9]\.[0-9]\.xyz
	rm -f A[0-9][0-9]\.[0-9]\.xyz
	rm -f initial.xyz
	rm -f M*\.txt
	rm -f core*
	rm -f _A[0-9]\.[0-9][0-9]\.xyz
	rm -f A[0-9]\.[0-9][0-9]\.xyz
	rm -f log.txt
	rm -f _initial.xyz

clean_dos :
	del mc_growth.exe *.o
