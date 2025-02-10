#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Функція, яку інтегруємо: f(x)=4/(1+x^2)
double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                   // Ініціалізація MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Отримання рангу процесу
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Кількість процесів

    // Межі інтегрування
    double a = 0.0, b = 1.0;
    // Кількість кроків
    long long num_steps = 1000000;
    double step = (b - a) / num_steps;
    double local_sum = 0.0;

    // Розподіл роботи між процесами
    long long steps_per_proc = num_steps / size;
    long long i_start = rank * steps_per_proc;
    long long i_end = (rank + 1) * steps_per_proc;
    if (rank == size - 1) {
        // Останній процес обробляє також і залишок
        i_end = num_steps;
    }

    double local_a = a + i_start * step;
    double local_b = a + i_end * step;
    printf("Процес %d: обробляє інтервал [%.6f, %.6f], індекси [%lld, %lld)\n",
           rank, local_a, local_b, i_start, i_end);

    // Обчислення локальної суми
    for (long long i = i_start; i < i_end; i++) {
        double x = a + i * step;
        local_sum += f(x);
    }
    printf("Процес %d: локальна сума = %.12f\n", rank, local_sum);

    // Збір локальних сум у процесі 0
    double total_sum = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double integral = total_sum * step;
        printf("\nОбчислений інтеграл: %.12f\n", integral);
        printf("Точне значення інтегралу (π): %.12f\n", M_PI);
    }

    MPI_Finalize();
    return 0;
}
