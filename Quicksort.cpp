#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 10000 // Size of the array to sort

// Function to swap two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to select a better pivot (median-of-three)
int choose_pivot(int arr[], int low, int high) {
    int mid = low + (high - low) / 2;
    // Sort low, mid, high values to get median
    if (arr[low] > arr[mid]) swap(&arr[low], &arr[mid]);
    if (arr[low] > arr[high]) swap(&arr[low], &arr[high]);
    if (arr[mid] > arr[high]) swap(&arr[mid], &arr[high]);
    // Place pivot at high
    swap(&arr[mid], &arr[high]);
    return arr[high];
}

// Function to partition the array
int partition(int arr[], int low, int high) {
    int pivot = choose_pivot(arr, low, high);
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

// Sequential Quicksort with loop prevention
void quicksort(int arr[], int low, int high) {
    if (low < high && low >= 0 && high < ARRAY_SIZE) { // Prevent invalid indices
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// Function to merge sorted subarrays
void merge(int arr[], int temp[], int low, int mid, int high) {
    int i = low, j = mid + 1, k = low;

    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= high) {
        temp[k++] = arr[j++];
    }

    for (i = low; i <= high; i++) {
        arr[i] = temp[i];
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    int *arr = NULL, *local_arr = NULL, *temp = NULL;
    int local_size, *sendcounts = NULL, *displs = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if ARRAY_SIZE is valid
    if (ARRAY_SIZE <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: ARRAY_SIZE must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate sendcounts and displacements for uneven distribution
    sendcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    local_size = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder) ? local_size + 1 : local_size;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    // Root process initializes the array
    if (rank == 0) {
        arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            arr[i] = rand() % 10000;
        }
        printf("Initial array (first 10 elements): ");
        for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    // Allocate local array for each process
    local_arr = (int*)malloc(sendcounts[rank] * sizeof(int));
    temp = (int*)malloc(ARRAY_SIZE * sizeof(int));

    // Start timing
    start_time = MPI_Wtime();

    // Scatter the array to all processes
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT, local_arr, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local array
    quicksort(local_arr, 0, sendcounts[rank] - 1);

    // Gather sorted subarrays back to root
    MPI_Gatherv(local_arr, sendcounts[rank], MPI_INT, arr, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process merges the sorted subarrays
    if (rank == 0) {
        // Copy gathered array to temp for merging
        for (int i = 0; i < ARRAY_SIZE; i++) {
            temp[i] = arr[i];
        }

        // Perform merge for all subarrays
        for (int step = 1; step < size; step *= 2) {
            for (int i = 0; i < size; i += 2 * step) {
                int low = displs[i];
                int mid = (i + step < size) ? displs[i + step] - 1 : ARRAY_SIZE - 1;
                int high = (i + 2 * step < size) ? displs[i + 2 * step] - 1 : ARRAY_SIZE - 1;

                if (mid >= ARRAY_SIZE || low >= high) continue;
                if (high >= ARRAY_SIZE) high = ARRAY_SIZE - 1;

                merge(arr, temp, low, mid, high);
            }
        }

        // End timing
        end_time = MPI_Wtime();

        // Print first 10 elements of sorted array
        printf("Sorted array (first 10 elements): ");
        for (int i = 0; i < 10 && i < ARRAY_SIZE; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        // Verify if array is sorted
        int is_sorted = 1;
        for (int i = 1; i < ARRAY_SIZE; i++) {
            if (arr[i - 1] > arr[i]) {
                is_sorted = 0;
                break;
            }
        }
        printf("Array is %s\n", is_sorted ? "sorted" : "not sorted");
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    // Clean up
    if (rank == 0) {
        free(arr);
    }
    free(local_arr);
    free(temp);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
