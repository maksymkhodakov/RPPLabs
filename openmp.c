// mandelbrot_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define WIDTH 3000
#define HEIGHT 3000
#define MAX_ITER 1000

int main(void) {
    int i, j;
    long long total = 0;       // контрольна сума (сума усіх ітерацій)
    int global_min = MAX_ITER; // мінімальне число ітерацій
    int global_max = 0;        // максимальне число ітерацій
    double start_time, end_time;

    // Виділення пам'яті для збереження значень ітерацій для кожного пікселя
    int *image = malloc(WIDTH * HEIGHT * sizeof(int));
    if (image == NULL) {
        fprintf(stderr, "Помилка виділення пам'яті!\n");
        return 1;
    }

    // Параметри області на комплексній площині
    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    start_time = omp_get_wtime();

    // Створюємо масив для зберігання кількості рядків, оброблених кожним потоком
    int max_threads = omp_get_max_threads();
    int *rows_processed = calloc(max_threads, sizeof(int));
    if (rows_processed == NULL) {
        fprintf(stderr, "Помилка виділення пам'яті для rows_processed!\n");
        free(image);
        return 1;
    }

    // Паралельне обчислення множини Мандельброта з логуванням роботи потоків
#pragma omp parallel for private(j) reduction(+:total) reduction(min:global_min) reduction(max:global_max) schedule(dynamic)
    for (i = 0; i < HEIGHT; i++) {
        int tid = omp_get_thread_num();
        rows_processed[tid]++;  // збільшуємо лічильник рядків для поточного потоку

        double imag = imag_max - i * (imag_max - imag_min) / (HEIGHT - 1);
        for (j = 0; j < WIDTH; j++) {
            double real = real_min + j * (real_max - real_min) / (WIDTH - 1);
            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;
            while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < MAX_ITER) {
                double temp = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0 * z_real * z_imag + imag;
                z_real = temp;
                iter++;
            }
            image[i * WIDTH + j] = iter;
            total += iter;
            if (iter < global_min)
                global_min = iter;
            if (iter > global_max)
                global_max = iter;
        }
        // Вивід логування кожного 500-го рядка (щоб не перевантажувати консоль)
        if (i % 500 == 0) {
#pragma omp critical
            {
                printf("Thread %d: оброблено рядок %d\n", tid, i);
            }
        }
    }

    end_time = omp_get_wtime();

    // Вивід інформації про роботу кожного потоку
    for (i = 0; i < max_threads; i++) {
        printf("Thread %d обробив %d рядків\n", i, rows_processed[i]);
    }

    // Вивід загальної статистики
    printf("\nOpenMP: Множина Мандельброта\n");
    printf("Контрольна сума (сума ітерацій): %lld\n", total);
    printf("Мінімальна кількість ітерацій: %d\n", global_min);
    printf("Максимальна кількість ітерацій: %d\n", global_max);
    printf("Час виконання: %f секунд\n", end_time - start_time);

    // Вивід декількох зразкових значень пікселів
    printf("Зразкові значення:\n");
    printf("Верхній лівий піксель: %d\n", image[0]);
    printf("Верхній правий піксель: %d\n", image[WIDTH - 1]);
    printf("Нижній лівий піксель: %d\n", image[(HEIGHT - 1) * WIDTH]);
    printf("Нижній правий піксель: %d\n", image[HEIGHT * WIDTH - 1]);
    printf("Центр: %d\n", image[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);

    free(image);
    free(rows_processed);
    return 0;
}
