#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 3000
#define HEIGHT 3000
#define MAX_ITER 1000

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Розподіл рядків між процесами
    int rows_per_proc = HEIGHT / size;
    int remainder = HEIGHT % size;
    int start_row, end_row;
    if (rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + remainder;
        end_row = start_row + rows_per_proc;
    }
    int local_rows = end_row - start_row;

    // Виділення пам'яті для локального блоку (масив значень ітерацій)
    int *local_image = malloc(local_rows * WIDTH * sizeof(int));
    if (local_image == NULL) {
        fprintf(stderr, "Процес %d: помилка виділення пам'яті\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long local_total = 0;
    int local_min = MAX_ITER;
    int local_max = 0;

    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    double start_time = MPI_Wtime();

    // Обчислення локального блоку
    for (int i = 0; i < local_rows; i++) {
        int global_row = start_row + i;
        // Вивід повідомлень кожного 500-го глобального рядка
        if (global_row % 500 == 0) {
            printf("Process %d: обробка глобального рядка %d\n", rank, global_row);
        }
        double imag = imag_max - global_row * (imag_max - imag_min) / (HEIGHT - 1);
        for (int j = 0; j < WIDTH; j++) {
            double real = real_min + j * (real_max - real_min) / (WIDTH - 1);
            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;
            while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < MAX_ITER) {
                double temp = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0 * z_real * z_imag + imag;
                z_real = temp;
                iter++;
            }
            local_image[i * WIDTH + j] = iter;
            local_total += iter;
            if (iter < local_min)
                local_min = iter;
            if (iter > local_max)
                local_max = iter;
        }
    }

    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    // Об'єднання локальних статистик із всіх процесів
    long long global_total;
    int global_min, global_max;
    double global_time;
    MPI_Reduce(&local_total, &global_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Вивід кількості оброблених рядків для кожного процесу
    printf("Process %d обробив %d рядків\n", rank, local_rows);

    // Збір повної матриці (з усіх процесів) на процесі 0
    int *global_image = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        global_image = malloc(HEIGHT * WIDTH * sizeof(int));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if (global_image == NULL || recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "Process 0: помилка виділення пам'яті для збору даних\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int p = 0; p < size; p++) {
            int p_rows = (p < remainder) ? (rows_per_proc + 1) : rows_per_proc;
            recvcounts[p] = p_rows * WIDTH;
            displs[p] = (p == 0) ? 0 : displs[p - 1] + recvcounts[p - 1];
        }
    }

    MPI_Gatherv(local_image, local_rows * WIDTH, MPI_INT,
                global_image, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Процес 0 виводить загальні результати та значення пікселів у кутах/центрі
    if (rank == 0) {
        printf("\nMPI: Множина Мандельброта\n");
        printf("Глобальна контрольна сума (сума ітерацій): %lld\n", global_total);
        printf("Глобально мінімальна кількість ітерацій: %d\n", global_min);
        printf("Глобально максимальна кількість ітерацій: %d\n", global_max);
        printf("Час виконання: %f секунд\n", global_time);
        // Вивід значень пікселів у кутах та центрі
        printf("Верхній лівий піксель: %d\n", global_image[0]);
        printf("Верхній правий піксель: %d\n", global_image[WIDTH - 1]);
        printf("Нижній лівий піксель: %d\n", global_image[(HEIGHT - 1) * WIDTH]);
        printf("Нижній правий піксель: %d\n", global_image[HEIGHT * WIDTH - 1]);
        printf("Центр: %d\n", global_image[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);
    }

    free(local_image);
    if (rank == 0) {
        free(global_image);
        free(recvcounts);
        free(displs);
    }
    MPI_Finalize();
    return 0;
}
