#include <omp.h>
#include <iostream>
#include <vector>
#include <limits>
#include <string>

// Function for parallel min reduction
float parallel_min(const std::vector<float>& input) {
    float global_min = std::numeric_limits<float>::max();
    #pragma omp parallel
    {
        float local_min = std::numeric_limits<float>::max();
        #pragma omp for
        for (size_t i = 0; i < input.size(); ++i) {
            local_min = std::min(local_min, input[i]);
        }
        #pragma omp critical
        {
            global_min = std::min(global_min, local_min);
        }
    }
    return global_min;
}

// Function for parallel max reduction
float parallel_max(const std::vector<float>& input) {
    float global_max = -std::numeric_limits<float>::max();
    #pragma omp parallel
    {
        float local_max = -std::numeric_limits<float>::max();
        #pragma omp for
        for (size_t i = 0; i < input.size(); ++i) {
            local_max = std::max(local_max, input[i]);
        }
        #pragma omp critical
        {
            global_max = std::max(global_max, local_max);
        }
    }
    return global_max;
}

// Function for parallel sum reduction
float parallel_sum(const std::vector<float>& input) {
    float global_sum = 0.0f;
    #pragma omp parallel reduction(+:global_sum)
    {
        #pragma omp for
        for (size_t i = 0; i < input.size(); ++i) {
            global_sum += input[i];
        }
    }
    return global_sum;
}

// Function to validate and get a float input
float get_float_input(const std::string& prompt) {
    float value;
    std::string input;
    while (true) {
        std::cout << prompt;
        std::getline(std::cin, input);
        try {
            value = std::stof(input);
            return value;
        } catch (const std::exception&) {
            std::cout << "Invalid input. Please enter a valid number.\n";
        }
    }
}

// Function to validate and get a positive integer input
size_t get_size_input(const std::string& prompt) {
    size_t value;
    std::string input;
    while (true) {
        std::cout << prompt;
        std::getline(std::cin, input);
        try {
            value = std::stoul(input);
            if (value > 0) {
                return value;
            } else {
                std::cout << "Please enter a positive number.\n";
            }
        } catch (const std::exception&) {
            std::cout << "Invalid input. Please enter a valid positive integer.\n";
        }
    }
}

int main() {
    // Get array size from user
    size_t N = get_size_input("Enter the number of elements: ");

    // Initialize input array
    std::vector<float> input(N);
    std::cout << "Enter " << N << " floating-point numbers:\n";
    for (size_t i = 0; i < N; ++i) {
        std::string prompt = "Element " + std::to_string(i + 1) + ": ";
        input[i] = get_float_input(prompt);
    }

    // Compute and display results
    if (N > 0) {
        float min_result = parallel_min(input);
        std::cout << "Minimum: " << min_result << std::endl;

        float max_result = parallel_max(input);
        std::cout << "Maximum: " << max_result << std::endl;

        float sum_result = parallel_sum(input);
        std::cout << "Sum: " << sum_result << std::endl;

        float avg_result = sum_result / static_cast<float>(N);
        std::cout << "Average: " << avg_result << std::endl;
    } else {
        std::cout << "Array is empty. No computations performed.\n";
    }

    return 0;
}
