#include "util.cuh"

template<typename T>
void printHostVector(const std::string name, const thrust::host_vector<T> &vec){
    int i = 0;
    std::cout <<"[" << name << "] -> ";
    for(auto &el : vec){
        std::cout << el << " ";
        i++;
        if(i > 40) break;
    }
    std::cout << "\n";
}

template<typename T>
__global__ void printDeviceVectorKernel(const T* d_vec, size_t size) {
    for (int i = 0; i < size; i++) {
        if constexpr (cuda::std::is_same<T, float>::value) {
            printf("%f ", d_vec[i]); // Use %f for float
        } else if constexpr (cuda::std::is_same<T, long>::value || cuda::std::is_same<T, unsigned long>::value) {
            printf("%ld ", d_vec[i]); // Use %ld for long
        } else if constexpr (cuda::std::is_same<T, int>::value) {
            printf("%d ", d_vec[i]); // Use %d for int
        } else {
            printf("Unsupported type\n");
            break;
        }
    }
}

template<typename T>
void printDeviceVector(const std::string name, const thrust::device_vector<T>& d_vec) {
    size_t size = d_vec.size() >= 40 ? 40 : d_vec.size();
    std::cout <<"[" << name << "] -> ";
    // Launch kernel to print device vector
    const T* d_vec_ptr = thrust::raw_pointer_cast(d_vec.data());

    // Call the kernel
    printDeviceVectorKernel<<<1, 1>>>(d_vec_ptr, size);

    // Synchronize to ensure all prints are complete
    cudaDeviceSynchronize();

    std::cout << "\n";
}

template void printHostVector<unsigned long>(const std::string name, const thrust::host_vector<unsigned long> &vec);
template void printDeviceVector<unsigned long>(const std::string name, const thrust::device_vector<unsigned long> &d_vec);
template void printDeviceVector<float>(const std::string name, const thrust::device_vector<float> &d_vec);

unsigned long computeNearestPowerOf2(unsigned long num)
{
    if (num == 0) return 1; // Handle the case where n is 0, since 0 doesn't have a meaningful power of two ceiling.

    // If n is already a power of two, return n.
    if ((num & (num - 1)) == 0) return num;

    // Use the bit trick to calculate the next power of two.
    num--;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    num |= num >> 32;
    return num + 1;
}
