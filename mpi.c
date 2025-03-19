#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int WIDTH, HEIGHT, MAX_ITER;
    double real_min, real_max, imag_min, imag_max;

    // Ініціалізація MPI-середовища.
    // Ця функція викликається першою, щоб підготувати MPI до роботи.
    MPI_Init(&argc, &argv);

    // Отримання унікального номера (rank) поточного процесу в комунікаторі MPI_COMM_WORLD.
    // Значення rank від 0 до size-1, де 0 — це "root" або "мастер".
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Отримання загальної кількості процесів, що беруть участь у виконанні програми.
    // Це дозволяє нам динамічно розподілити завдання між процесами.
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Процес з rank 0 задає вхідні параметри (hardcoded).
    // Інші процеси отримають ці значення через MPI_Bcast.
    if (rank == 0) {
        WIDTH = 3000;
        HEIGHT = 3000;
        MAX_ITER = 1000;
        real_min = -2.0;
        real_max = 1.0;
        imag_min = -1.5;
        imag_max = 1.5;
    }

    // Розсилка вхідних параметрів від процесу 0 (root) до всіх інших процесів.
    // MPI_Bcast (broadcast) забезпечує, що всі процеси отримають однакові дані.
    MPI_Bcast(&WIDTH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&HEIGHT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_ITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&real_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&real_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imag_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imag_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Після цих викликів, усі процеси мають однакові значення параметрів.

    // Розподіл рядків зображення між процесами.
    // Кожен процес отримає певну кількість рядків для обчислення.
    int rows_per_proc = HEIGHT / size;
    int remainder = HEIGHT % size;
    int start_row, end_row;
    // Якщо rank менший за залишок, процесу виділяється додатковий рядок.
    if (rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + remainder;
        end_row = start_row + rows_per_proc;
    }
    int local_rows = end_row - start_row;

    // Виділення пам'яті для локального блоку зображення.
    // Кожен процес створює масив, в якому зберігатиме результати обчислень (кількість ітерацій) для своїх рядків.
    int *local_image = malloc(local_rows * WIDTH * sizeof(int));

    // Локальні статистичні змінні для накопичення результатів:
    // - local_total: сума ітерацій для локального блоку;
    // - local_min: мінімальна кількість ітерацій серед пікселів локального блоку;
    // - local_max: максимальна кількість ітерацій.
    long long local_total = 0;
    int local_min = MAX_ITER;
    int local_max = 0;

    // Фіксуємо час початку локальних обчислень.
    // MPI_Wtime повертає поточний час, що дозволяє нам обчислити час виконання.
    double start_time = MPI_Wtime();

    // Обчислення локального блоку (рядків) множини Мандельброта.
    // Кожен процес обчислює свій набір пікселів, що відповідають його діапазону рядків.
    for (int i = 0; i < local_rows; i++) {
        int global_row = start_row + i;
        // Для кожного 500-го глобального рядка виводимо інформацію про хід обчислень.
        if (global_row % 500 == 0) {
            printf("Процес %d: обробка глобального рядка %d\n", rank, global_row);
        }
        double imag = imag_max - global_row * (imag_max - imag_min) / (HEIGHT - 1);
        for (int j = 0; j < WIDTH; j++) {
            double real = real_min + j * (real_max - real_min) / (WIDTH - 1);
            double z_real = 0.0, z_imag = 0.0;
            int iter = 0;
            // Основний цикл обчислення для визначення, чи належить точка множині Мандельброта.
            while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < MAX_ITER) {
                double temp = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0 * z_real * z_imag + imag;
                z_real = temp;
                iter++;
            }
            // Збереження результату для пікселя.
            local_image[i * WIDTH + j] = iter;
            local_total += iter;
            if (iter < local_min)
                local_min = iter;
            if (iter > local_max)
                local_max = iter;
        }
    }


    long long global_total;
    int global_min, global_max;
    double global_time;

    // Збір глобальної статистики за допомогою MPI_Reduce.
    // MPI_Reduce виконує операцію редукції (сума, мінімум, максимум) над значеннями з усіх процесів.
    // Результат операції надсилається до процесу, заданого як root (тут 0).
    MPI_Reduce(&local_total, &global_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Фіксуємо час завершення обчислень разом з кінцем передачі усіх даних на 0 процес
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    // Для часу обчислень беремо максимальне значення серед усіх процесів,
    // що дає нам загальний час виконання, оскільки деякі процеси можуть працювати довше.
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Підготовка до збору глобального зображення.
    // Процес 0 (root) виділяє пам'ять для отримання результатів з усіх процесів.
    // Інші процеси також виділяють пам'ять для прийому даних, що будуть розіслані через MPI_Bcast.
    int *global_image = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        global_image = malloc(HEIGHT * WIDTH * sizeof(int));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        // Обчислюємо, скільки елементів (пікселів) має приймати кожен процес,
        // а також зсув (displacement) для кожного процесу в глобальному масиві.
        for (int p = 0; p < size; p++) {
            int p_rows = (p < remainder) ? (rows_per_proc + 1) : rows_per_proc;
            recvcounts[p] = p_rows * WIDTH;
            displs[p] = (p == 0) ? 0 : displs[p - 1] + recvcounts[p - 1];
        }
    } else {
        // На процесах, що не є root, також виділяємо пам'ять для global_image,
        // щоб отримати результат розсилки через MPI_Bcast.
        global_image = malloc(HEIGHT * WIDTH * sizeof(int));
    }

    // Збір локальних результатів у глобальний масив на процесі 0.
    // MPI_Gatherv дозволяє збирати змінну кількість елементів від кожного процесу.
    // На non-root процесах send buffer (local_image) надсилається, а на root він приймається і зберігається в global_image.
    MPI_Gatherv(local_image, local_rows * WIDTH, MPI_INT,
                global_image, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Лише процес 0 виводить підсумкову інформацію та статистику обчислень.
    if (rank == 0) {
        printf("\nMPI: Множина Мандельброта\n");
        printf("Глобальна контрольна сума (сума ітерацій): %lld\n", global_total);
        printf("Глобально мінімальна кількість ітерацій: %d\n", global_min);
        printf("Глобально максимальна кількість ітерацій: %d\n", global_max);
        printf("Час виконання: %f секунд\n", global_time);
        // Вивід зразкових значень пікселів (кутів та центру)
        printf("Верхній лівий піксель: %d\n", global_image[0]);
        printf("Верхній правий піксель: %d\n", global_image[WIDTH - 1]);
        printf("Нижній лівий піксель: %d\n", global_image[(HEIGHT - 1) * WIDTH]);
        printf("Нижній правий піксель: %d\n", global_image[HEIGHT * WIDTH - 1]);
        printf("Центр: %d\n", global_image[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);
    }

    // Звільнення виділеної пам'яті.
    free(local_image);
    free(global_image);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    // Завершення роботи MPI-середовища.
    // Після виклику MPI_Finalize більше не можна використовувати MPI-функції.
    MPI_Finalize();
    return 0;
}
