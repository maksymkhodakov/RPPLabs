#include <stdio.h>
#include <stdlib.h>
#include <omp.h>    // Підключення бібліотеки OpenMP для паралельного програмування
#include <math.h>

#define WIDTH 3000       // Ширина зображення
#define HEIGHT 3000      // Висота зображення
#define MAX_ITER 1000    // Максимальна кількість ітерацій для обчислення пікселя
#define NUM_FRAMES 10   // Кількість кадрів для анімації

int main(void) {
    int frame, i, j;
    // Центр області, на яку будемо збільшувати (zoom)
    double center_real = -0.5, center_imag = 0.0;
    // Початковий розмір області (ширина та висота у комплексній площині)
    double initial_width = 3.0;
    double initial_height = 3.0;
    // Крок збільшення: для кожного наступного кадру збільшення відбувається за цією базою
    double zoom_step = 1.05;

    // Генерація NUM_FRAMES кадрів для динамічної візуалізації
    for (frame = 0; frame < NUM_FRAMES; frame++) {
        // Обчислення коефіцієнта збільшення для поточного кадру
        double zoom = pow(zoom_step, frame);
        // Обчислення поточного розміру області (зменшується при збільшенні)
        double current_width = initial_width / zoom;
        double current_height = initial_height / zoom;
        // Обчислення меж області для побудови множини Мандельброта
        double real_min = center_real - current_width / 2;
        double real_max = center_real + current_width / 2;
        double imag_min = center_imag - current_height / 2;
        double imag_max = center_imag + current_height / 2;

        // Ініціалізація глобальних змінних для статистики обчислень
        long long total = 0;       // Контрольна сума: сума ітерацій для всіх пікселів
        int global_min = MAX_ITER; // Змінна для збереження мінімальної кількості ітерацій для пікселя
        int global_max = 0;        // Змінна для збереження максимальної кількості ітерацій для пікселя

        // Виділення пам'яті для збереження значень ітерацій для кожного пікселя
        int *image = malloc(WIDTH * HEIGHT * sizeof(int));
        if (image == NULL) {
            fprintf(stderr, "Помилка виділення пам'яті для кадру %d!\n", frame);
            return 1;
        }

        // Отримання стартового часу для обчислення кадру
        double start_time = omp_get_wtime();

        // Отримання максимальної кількості потоків, які можуть бути запущені
        int max_threads = omp_get_max_threads();
        // Виділення пам'яті для масиву, що зберігає кількість оброблених рядків кожним потоком
        int *rows_processed = calloc(max_threads, sizeof(int));
        if (rows_processed == NULL) {
            fprintf(stderr, "Помилка виділення пам'яті для rows_processed в кадрі %d!\n", frame);
            free(image);
            return 1;
        }

        // Паралельна область: розрахунок множини Мандельброта для поточного кадру
        // Збережено оригінальну логіку OpenMP:
        // - private(j): кожному потоку виділяється власна копія змінної j
        // - reduction(+:total): акумуляція загальної суми ітерацій
        // - reduction(min:global_min): пошук мінімального значення
        // - reduction(max:global_max): пошук максимального значення
        // - schedule(dynamic): динамічний розподіл ітерацій між потоками
#pragma omp parallel for private(j) reduction(+:total) reduction(min:global_min) reduction(max:global_max) schedule(dynamic)
        for (i = 0; i < HEIGHT; i++) {
            // Отримання ідентифікатора поточного потоку
            int tid = omp_get_thread_num();
            // Лічильник, що фіксує, скільки рядків обробив даний потік
            rows_processed[tid]++;

            // Обчислення значення уявної частини для поточного рядка
            double imag_val = imag_max - i * (imag_max - imag_min) / (HEIGHT - 1);
            for (j = 0; j < WIDTH; j++) {
                // Обчислення значення дійсної частини для поточного стовпця
                double real_val = real_min + j * (real_max - real_min) / (WIDTH - 1);
                double z_real = 0.0, z_imag = 0.0;
                int iter = 0;
                // Основний цикл розрахунку ітерацій для пікселя:
                // Перевіряємо, чи не вийшла точка за межі кола радіусом 2
                while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < MAX_ITER) {
                    double temp = z_real * z_real - z_imag * z_imag + real_val;
                    z_imag = 2.0 * z_real * z_imag + imag_val;
                    z_real = temp;
                    iter++;
                }
                // Запис результату для поточного пікселя у відповідну позицію масиву image
                image[i * WIDTH + j] = iter;
                // Оновлення глобальних статистичних значень
                total += iter;
                if (iter < global_min)
                    global_min = iter;
                if (iter > global_max)
                    global_max = iter;
            }
            // Вивід логування кожного 500-го рядка для моніторингу роботи потоків
            if (i % 500 == 0) {
                printf("Frame %d, Thread %d: оброблено рядок %d\n", frame, tid, i);
            }
        }

        // Фіксація кінцевого часу обчислення кадру
        double end_time = omp_get_wtime();
        printf("Frame %d generated in %f секунд. total = %lld, min = %d, max = %d\n",
               frame, end_time - start_time, total, global_min, global_max);

        // Вивід інформації про кількість рядків, оброблених кожним потоком (опціонально)
        for (i = 0; i < max_threads; i++) {
            printf("Frame %d: Thread %d обробив %d рядків\n", frame, i, rows_processed[i]);
        }
        free(rows_processed);

        // Формування імені файлу для поточного кадру (наприклад, frame_0000.ppm, frame_0001.ppm, ...)
        char filename[256];
        sprintf(filename, "frame_%04d.ppm", frame);

        // Відкриття файлу для запису зображення у форматі PPM (бінарний режим)
        FILE *fp = fopen(filename, "wb");
        if (fp == NULL) {
            fprintf(stderr, "Помилка відкриття файлу %s для запису зображення!\n", filename);
            free(image);
            return 1;
        }
        // Запис заголовку файлу PPM: формат P6, розміри зображення та максимальне значення кольору (255)
        fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

        // Перетворення значень ітерацій у відтінки сірого та запис пікселів у файл
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                int iter = image[i * WIDTH + j];
                // Масштабування ітерацій до діапазону 0-255
                unsigned char color = (unsigned char)(255 * iter / MAX_ITER);
                // Запис значення для трьох каналів (R, G, B)
                fwrite(&color, 1, 1, fp);
                fwrite(&color, 1, 1, fp);
                fwrite(&color, 1, 1, fp);
            }
        }
        // Закриття файлу після завершення запису кадру
        fclose(fp);

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

        // Звільнення пам'яті, виділеної для поточного кадру
        free(image);
    }

    // Після генерації всіх кадрів об'єднання їх у анімований GIF за допомогою ImageMagick.
    // Команда 'convert' бере всі файли frame_*.ppm, встановлює затримку між кадрами (-delay 10)
    // та нескінченну петлю (-loop 0), створюючи GIF файл animation.gif.
    system("convert -delay 10 -loop 0 frame_*.ppm animation.gif");
    printf("Animated GIF generated as animation.gif\n");

    return 0;
}
