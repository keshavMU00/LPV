from mpi4py import MPI
import numpy as np
import time
import platform
import statistics

def sequential_quicksort(arr, start, end):
    """Sequential Quicksort implementation."""
    if start < end:
        pivot = arr[end]
        i = start - 1
        for j in range(start, end + 1):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        pi = i
        sequential_quicksort(arr, start, pi - 1)
        sequential_quicksort(arr, pi + 1, end)

def merge_sorted_arrays(arr1, arr2):
    """Merge two sorted arrays."""
    result = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

def parallel_quicksort(comm, rank, size, data, n):
    """Parallel Quicksort with accurate timing."""
    if size == 1:
        start_comp = time.perf_counter()
        local_data = data.copy()
        sequential_quicksort(local_data, 0, n - 1)
        comp_time = time.perf_counter() - start_comp
        return local_data.tolist(), comp_time, 0.0

    chunk_size = n // size
    if rank == size - 1:
        chunk_size = n - (size - 1) * chunk_size

    local_data = np.zeros(chunk_size, dtype=np.int32)
    
    comm_time = 0
    start_comm = time.perf_counter()
    comm.Scatter(data, local_data, root=0)
    comm_time += time.perf_counter() - start_comm

    start_comp = time.perf_counter()
    sequential_quicksort(local_data, 0, len(local_data) - 1)
    comp_time = time.perf_counter() - start_comp

    start_comm = time.perf_counter()
    if rank == 0:
        sorted_chunks = [local_data]
        for i in range(1, size):
            recv_chunk = np.zeros(n // size if i < size - 1 else n - (size - 1) * (n // size), dtype=np.int32)
            comm.Recv(recv_chunk, source=i)
            sorted_chunks.append(recv_chunk)
        result = sorted_chunks[0].tolist()
        for i in range(1, len(sorted_chunks)):
            result = merge_sorted_arrays(result, sorted_chunks[i].tolist())
    else:
        comm.Send(local_data, dest=0)
        result = None
    comm_time += time.perf_counter() - start_comm

    return result, comp_time, comm_time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"System: {platform.processor()}, {platform.system()} {platform.release()}")
        print(f"MPI Version: {MPI.Get_library_version().splitlines()[0]}")
        print(f"Number of processes: {size}")

    num_iterations = 10  # Increased for reliability

    if rank == 0:
        try:
            num_sizes = int(input("Enter the number of different array sizes to test: "))
            if num_sizes <= 0:
                print("Please enter a positive number.")
                comm.Abort()
            n_values = []
            for _ in range(num_sizes):
                n = int(input("Enter an array size: "))
                if n <= 0:
                    print("Please enter a positive number.")
                    comm.Abort()
                n_values.append(n)
        except ValueError:
            print("Error: Array size must be an integer.")
            comm.Abort()
    else:
        num_sizes = None
        n_values = None

    num_sizes = comm.bcast(num_sizes, root=0)
    if rank == 0:
        n_values = n_values
    else:
        n_values = [None] * num_sizes
    n_values = comm.bcast(n_values, root=0)

    for n in n_values:
        if rank == 0:
            try:
                print(f"\nTesting array size: {n}")
                print(f"Enter {n} space-separated integers for the array:")
                array_input = input().strip().split()
                if len(array_input) != n:
                    print(f"Error: Expected {n} elements, but got {len(array_input)}.")
                    comm.Abort()
                data = [int(x) for x in array_input]
                data = np.array(data, dtype=np.int32)
                print(f"Input array: {data.tolist()}" if n <= 10 else f"Input array size: {n}")
            except ValueError:
                print("Error: All array elements must be integers.")
                comm.Abort()
        else:
            data = None

        if rank == 0:
            data = np.array(data, dtype=np.int32)
        else:
            data = np.zeros(n, dtype=np.int32)
        comm.Bcast(data, root=0)

        seq_times = []
        if rank == 0:
            for _ in range(num_iterations):
                seq_data = data.copy()
                start_time = time.perf_counter()
                sequential_quicksort(seq_data, 0, n - 1)
                seq_times.append(time.perf_counter() - start_time)
                if not np.all(np.diff(seq_data) >= 0):
                    print("Sequential sort failed!")
                if n <= 10:
                    print(f"Sequential sorted array: {seq_data.tolist()}")
            avg_seq_time = statistics.mean(seq_times)
            std_seq_time = statistics.stdev(seq_times) if num_iterations > 1 else 0
            print(f"Sequential Quicksort time (avg ± std): {avg_seq_time:.6f} ± {std_seq_time:.6f} seconds")

        par_times = []
        comp_times = []
        comm_times = []
        for _ in range(num_iterations):
            comm.Barrier()
            start_time = time.perf_counter()
            sorted_data, comp_time, comm_time = parallel_quicksort(comm, rank, size, data, n)
            comm.Barrier()
            par_times.append(time.perf_counter() - start_time)
            comp_times.append(comp_time)
            comm_times.append(comm_time)

        all_comp_times = comm.gather(comp_times, root=0)

        if rank == 0:
            avg_par_time = statistics.mean(par_times)
            std_par_time = statistics.stdev(par_times) if num_iterations > 10 else 0
            avg_comm_time = statistics.mean(comm_times)
            avg_comp_time = statistics.mean(comp_times)
            overhead_time = avg_par_time - (avg_comm_time + avg_comp_time)
            print(f"Parallel Quicksort time (avg ± std, {size} processes): {avg_par_time:.6f} ± {std_par_time:.6f} seconds")
            print(f"Communication time: {avg_comm_time:.6f} seconds ({avg_comm_time/avg_par_time*100:.1f}%)")
            print(f"Computation time: {avg_comp_time:.6f} seconds ({avg_comp_time/avg_par_time*100:.1f}%)")
            print(f"Overhead time: {overhead_time:.6f} seconds ({overhead_time/avg_par_time*100:.1f}%)")
            
            flat_comp_times = [t for iter_times in all_comp_times for t in iter_times]
            max_comp = max(flat_comp_times)
            min_comp = min(flat_comp_times)
            avg_comp = statistics.mean(flat_comp_times)
            print(f"Load balance - Max comp time: {max_comp:.6f}s, Min: {min_comp:.6f}s, Avg: {avg_comp:.6f}s")

            if sorted_data is not None and not np.all(np.diff(np.array(sorted_data)) >= 0):
                print("Parallel sort failed!")
            if n <= 10:
                print(f"Parallel sorted array: {sorted_data}")

            if avg_par_time > 0:
                speedup = avg_seq_time / avg_par_time
                efficiency = speedup / size
                print(f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}")
            else:
                print("Speedup not calculated due to zero parallel time.")

if __name__ == "__main__":
    main()
