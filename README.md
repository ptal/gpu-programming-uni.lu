# GPU Programming Course @University of Luxembourg

Exercises for the course GPU Programming @University of Luxembourg

## Introduction to CUDA Programming (`1-introduction-cuda`)

We overview the CUDA model of computation.

* `1-introduction-cuda/demo`: contains the fully functional code demo shown and explained during the course.
* `1-introduction-cuda/exercises`: contains the exercises (`.cu` files).
* `1-introduction-cuda/solutions` contains the solutions of the exercises.

### BLAS `saxpy` [`2-cuda/saxpy`]

We reimplement the saxpy routine of the BLAS (Basic Linear Algebra Subproblems) library that is widely used (and heavily optimized) on many systems.
`saxpy` computes the simple operation `result = scale*X+Y`, where `X`, `Y`, and `result` are vectors of `N` elements (where `N` can be 20 millions in this project) and `scale` is a scalar.
Note that `saxpy` performs two math operations (one multiply, one add) for every three elements used.
`saxpy` is a *trivially parallelizable computation* and features predictable, regular data access and predictable execution cost.

1. Implement a basic sequential version of that program with time measurement.
```cpp
void saxpy(float scale, const std::vector<float>& X, const std::vector<float>& Y, std::vector<float>& result);
```
2. Implement a CUDA kernel parallelizing this operation, by paying attention to memory accesses and using managed memory for simplicity. Measure the time and compare the efficiency. At the end of the computation, verify you get correct results by comparing with the results provided by the sequential implementation.
3. Use global memory instead of managed memory. For that, you need to allocate the memory using `cudaMalloc`, followed by a `cudaMemcpy` to transfer the data from the CPU to the GPU, and finally by calling your kernel.
4. Measure the time taken by your kernel excluding and including the memory transfer.
5. [Optional⭐] Set up a CMake project inspired by `2-cuda/floyd`.

### Atomic Floyd Warshall [`2-cuda/floyd`]

* Modify Floyd Warshall to use `battery::vector<int, battery::managed_allocator>` instead of `int**` (check [this tutorial](https://lattice-land.github.io/2-cuda-battery.html)).
*  [Optional⭐⭐] Due to torn read/write (see lecture 3 of "parallel and grid computing"), our implementation of Floyd Warshall is, in theory, incorrect.
Wrap the integers in an atomic:
```c++
namespace bt = battery;
using atomic_int = bt::atomic_memory_grid::atomic_type<int>;
bt::vector<atomic_int, bt::global_allocator> v(10);
```
Of course, the vector cannot be allocated on the CPU since it is using `global_allocator`, hence you must initialized it in a kernel first through a `unique_ptr` (you can use the vector already declared in managed memory to initialize it).
Measure the performance with atomics.
* Think about another way to avoid the usage of atomics.

### Acknowledgments

Some exercises are inspired from the course [Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall23/) at Stanford and used with the permission of Kayvon Fatahalian.
Thanks to him and his teaching team!
