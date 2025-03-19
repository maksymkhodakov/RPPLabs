#include <stdio.h>
#include <stdlib.h>
#include <omp.h>    // Підключення бібліотеки OpenMP для паралельного програмування
#include <math.h>

#define WIDTH 3000
#define HEIGHT 3000
#define MAX_ITER 1000

int main(void) {
    int i, j;
    long long total = 0;       // Контрольна сума: сума ітерацій для всіх пікселів
    int global_min = MAX_ITER; // Змінна для збереження мінімальної кількості ітерацій, необхідних для розрахунку пікселя
    int global_max = 0;        // Змінна для збереження максимальної кількості ітерацій для пікселя
    double start_time, end_time;

    // Виділення пам'яті для збереження значень ітерацій для кожного пікселя
    int *image = malloc(WIDTH * HEIGHT * sizeof(int));
    if (image == NULL) {
        fprintf(stderr, "Помилка виділення пам'яті!\n");
        return 1;
    }

    // Параметри області на комплексній площині, для побудови множини Мандельброта
    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    // Отримання стартового часу за допомогою OpenMP (функція omp_get_wtime повертає поточний час)
    start_time = omp_get_wtime();

    // Отримання максимальної кількості потоків, які можуть бути запущені
    int max_threads = omp_get_max_threads();
    // Виділення пам'яті для масиву, що зберігає кількість оброблених рядків кожним потоком
    int *rows_processed = calloc(max_threads, sizeof(int));
    if (rows_processed == NULL) {
        fprintf(stderr, "Помилка виділення пам'яті для rows_processed!\n");
        free(image);
        return 1;
    }

    // Паралельна область: розрахунок множини Мандельброта
    // Директива omp parallel for створює паралельний цикл, розподіляючи ітерації циклу між потоками
    // private(j) - кожному потоку виділяється власна копія змінної j
    // reduction(+:total) - обчислення суми (total) з усіх потоків із сумуванням значень
    // reduction(min:global_min) - знаходження мінімального значення серед усіх потоків
    // reduction(max:global_max) - знаходження максимального значення серед усіх потоків
    // schedule(dynamic) - динамічний розподіл ітерацій циклу між потоками для балансування навантаження
#pragma omp parallel for private(j) reduction(+:total) reduction(min:global_min) reduction(max:global_max) schedule(dynamic)
    for (i = 0; i < HEIGHT; i++) {
        // Отримання ідентифікатора поточного потоку
        int tid = omp_get_thread_num();
        // Лічильник, що фіксує, скільки рядків обробив даний потік
        rows_processed[tid]++;

        // Обчислення значення уявної частини відповідного рядка
        double imag = imag_max - i * (imag_max - imag_min) / (HEIGHT - 1);
        for (j = 0; j < WIDTH; j++) {
            // Обчислення значення дійсної частини для даного стовпця
            double real = real_min + j * (real_max - real_min) / (WIDTH - 1);
            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;
            // Основний цикл розрахунку ітерацій для пікселя: перевіряється, чи не вийшла точка за межі кола радіусом 2
            while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < MAX_ITER) {
                double temp = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0 * z_real * z_imag + imag;
                z_real = temp;
                iter++;
            }
            // Запис результату для поточного пікселя у відповідну позицію масиву image
            image[i * WIDTH + j] = iter;
            // Оновлення загальної контрольної суми, мінімального та максимального значення ітерацій
            total += iter;
            if (iter < global_min)
                global_min = iter;
            if (iter > global_max)
                global_max = iter;
        }
        // Вивід логування кожного 500-го рядка для моніторингу роботи потоків
        if (i % 500 == 0) {
            printf("Thread %d: оброблено рядок %d\n", tid, i);
        }
    }

    // Фіксація кінцевого часу виконання
    end_time = omp_get_wtime();

    // Вивід інформації про кількість рядків, оброблених кожним потоком
    for (i = 0; i < max_threads; i++) {
        printf("Thread %d обробив %d рядків\n", i, rows_processed[i]);
    }

    // Вивід загальної статистики обчислень
    printf("\nOpenMP: Множина Мандельброта\n");
    printf("Контрольна сума (сума ітерацій): %lld\n", total);
    printf("Мінімальна кількість ітерацій: %d\n", global_min);
    printf("Максимальна кількість ітерацій: %d\n", global_max);
    printf("Час виконання: %f секунд\n", end_time - start_time);

    // Вивід декількох зразкових значень пікселів для перевірки
    printf("Зразкові значення:\n");
    printf("Верхній лівий піксель: %d\n", image[0]);
    printf("Верхній правий піксель: %d\n", image[WIDTH - 1]);
    printf("Нижній лівий піксель: %d\n", image[(HEIGHT - 1) * WIDTH]);
    printf("Нижній правий піксель: %d\n", image[HEIGHT * WIDTH - 1]);
    printf("Центр: %d\n", image[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);

    // Звільнення виділеної пам'яті
    free(image);
    free(rows_processed);
    return 0;
}
