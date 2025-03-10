#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 3000
#define HEIGHT 3000
#define MAX_ITER 1000

int main(void) {
    int i, j;
    long long total = 0;       // контрольна сума (сума ітерацій)
    int global_min = MAX_ITER; // мінімальна кількість ітерацій
    int global_max = 0;        // максимальна кількість ітерацій

    // Виділення пам'яті для збереження значень ітерацій для кожного пікселя
    int *image = malloc(WIDTH * HEIGHT * sizeof(int));
    if (image == NULL) {
        fprintf(stderr, "Помилка виділення пам'яті!\n");
        return 1;
    }

    // Параметри області на комплексній площині
    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    // Вимірювання часу виконання
    clock_t start_time = clock();

    // Обчислення множини Мандельброта (послідовно)
    for (i = 0; i < HEIGHT; i++) {
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
        // Логування кожного 500-го рядка
        if (i % 500 == 0) {
            printf("Оброблено рядок %d\n", i);
        }
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Вивід загальної статистики
    printf("\nСинхронна (послідовна) версія множини Мандельброта\n");
    printf("Контрольна сума (сума ітерацій): %lld\n", total);
    printf("Мінімальна кількість ітерацій: %d\n", global_min);
    printf("Максимальна кількість ітерацій: %d\n", global_max);
    printf("Час виконання: %f секунд\n", elapsed_time);

    // Вивід зразкових значень пікселів
    printf("\nЗразкові значення пікселів:\n");
    printf("Верхній лівий піксель: %d\n", image[0]);
    printf("Верхній правий піксель: %d\n", image[WIDTH - 1]);
    printf("Нижній лівий піксель: %d\n", image[(HEIGHT - 1) * WIDTH]);
    printf("Нижній правий піксель: %d\n", image[HEIGHT * WIDTH - 1]);
    printf("Центр: %d\n", image[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);

    free(image);
    return 0;
}
