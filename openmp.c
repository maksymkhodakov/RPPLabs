#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Функція, яку інтегруємо: f(x)=4/(1+x^2)
double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main() {
    // Межі інтегрування
    double a = 0.0, b = 1.0;
    // Кількість кроків (розбиття інтервалу)
    long long num_steps = 1000000;
    double step = (b - a) / num_steps;
    double total_sum = 0.0;

    printf("Обчислення інтегралу f(x)=4/(1+x^2) від %.2f до %.2f методом трапецій\n\n", a, b);

    // Паралельний регіон OpenMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        double local_sum = 0.0;

        // Розподіл ітерацій між потоками:
        // кожен потік обробляє свій підінтервал [i_start, i_end)
        long long i_start = tid * (num_steps / num_threads);
        long long i_end = (tid + 1) * (num_steps / num_threads);
        // Останній потік бере залишок (якщо num_steps не ділиться рівномірно)
        if (tid == num_threads - 1) {
            i_end = num_steps;
        }

        double local_a = a + i_start * step;
        double local_b = a + i_end * step;

        printf("Потік %d: обробляє інтервал [%.6f, %.6f], індекси [%lld, %lld)\n",
               tid, local_a, local_b, i_start, i_end);

        // Обчислення локальної суми
        // (Тут ми сумуємо значення f(x) для кожного кроку,
        //  а множення на step виконаємо пізніше)
        for (long long i = i_start; i < i_end; i++) {
            double x = a + i * step;
            local_sum += f(x);
        }
        printf("Потік %d: локальна сума = %.12f\n", tid, local_sum);

        // Додавання локального результату до загальної суми (з синхронізацією)
#pragma omp atomic
        total_sum += local_sum;
    } // завершення паралельного регіону

    double integral = total_sum * step;
    printf("\nОбчислений інтеграл: %.12f\n", integral);
    printf("Точне значення інтегралу (π): %.12f\n", M_PI);

    return 0;
}
