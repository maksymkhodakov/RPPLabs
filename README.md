# Розподілене та паралельне програмування  
# Ходаков Максим ТТП-42  
## Задача: Множина Мандельброта

# Інструкція до послідовного:

### gcc -o seq seq.c -lm
### ./seq

# Інструкція до OpenMP:  

### export OMP_NUM_THREADS=4
### gcc-14 -fopenmp openmp.c -o openmp
### ./openmp  

# Інструкція до MPI:

### mpicc mpi.c -o mpi
### mpirun -np 4 ./mpi

## На MacOS (на Apple Silicon) в разі помилки компіляції виконати
### brew reinstall open-mpi
### export PATH=/opt/homebrew/bin:$PATH
### which mpicc    

### Директорія Homebrew (/homebrew) є дефолтною для M1/M2 MacOS
