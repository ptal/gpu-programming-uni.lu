# Project: Parallel Prefix-Sum

## Important Information

### Rules

* Send the code and report in "name.surname.prefixsum.zip" before the next course begins (6th November 13h59) by email with the subject "[MHPC][GP] Prefix Sum Project".
* This project accounts for 5% of the total grade.
* Don't share your code.
* This is a solo project.
* The code provided is C-ished, not the beautiful C++11 (and above) standard we have seen previously. You must get trained to read the code of others, which is not necessarily beautiful.

## Project

Now that you're familiar with the basic structure and layout of CUDA programs, you are asked to come up with parallel implementation of the function `find_repeats` which, given a list of integers `A`, returns a list of all indices `i` for which `A[i] == A[i+1]`.

For example, given the array `{1,2,2,1,1,1,3,5,3,3}`, your program should output the array `{1,3,4,8}`.

#### Exclusive Prefix Sum

We want you to implement `find_repeats` by first implementing parallel exclusive prefix-sum operation.

Exlusive prefix sum takes an array `A` and produces a new array `output` that has, at each index `i`, the sum of all elements up to but not including `A[i]`. For example, given the array `A={1,4,6,8,2}`, the output of exclusive prefix sum `output={0,1,5,11,19}`.

The following "C-like" code is an iterative version of scan. In the pseudocode before, we use `parallel_for` to indicate potentially parallel loops. This is the same algorithm we discussed in class: <http://cs149.stanford.edu/fall23/lecture/dataparallel/slide_17>

```c++
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

We would like you to use this algorithm to implement a version of parallel prefix sum in CUDA. You must implement `exclusive_scan` function in `scan/scan.cu`. Your implementation will consist of both host and device code. The implementation will require multiple CUDA kernel launches (one for each parallel_for loop in the pseudocode above).

**Note:** In the starter code, the reference solution scan implementation above assumes that the input array's length (`N`) is a power of 2. In the `cudaScan` function, we solve this problem by rounding the input array length to the next power of 2 when allocating the corresponding buffers on the GPU.  However, the code only copies back `N` elements from the GPU buffer back to the CPU buffer.  This fact should simplify your CUDA implementation.

Compilation produces the binary `cudaScan`.  Commandline usage is as follows:
~~~~
Usage: ./cudaScan [options]

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
~~~~

#### Implementing "Find Repeats" Using Prefix Sum

Once you have written `exclusive_scan`, implement the function `find_repeats` in `scan/scan.cu`. This will involve writing more device code, in addition to one or more calls to `exclusive_scan()`. Your code should write the list of repeated elements into the provided output pointer (in device memory), and then return the size of the output list.

When calling your `exclusive_scan` implementation, remember that the contents of the `start` array are copied over to the `output` array.  Also, the arrays passed to `exclusive_scan` are assumed to be in `device` memory.

**Grading:** We will test your code for correctness and performance on random input arrays.

For reference, a scan score table is provided below, showing the performance of a simple CUDA implementation on a K80 GPU.  To check the correctness and performance score of your `scan` and `find_repeats` implementation, run **`./checker.pl scan`** and **`./checker.pl find_repeats`** respectively.  Doing so will produce a reference table as shown below; your score is based solely on the performance of your code.  In order to get full credit, your code must perform within 20% of the provided reference solution.

~~~~
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
~~~~

This part of the assignment is largely about getting more practice with writing CUDA and thinking in a data parallel manner, and not about performance tuning code. Getting full performance points on this part of the assignment should not require much (or really any) performance tuning, just a direct port of the algorithm pseudocode to CUDA.
However, there's one trick: a naive implementation of scan might launch N CUDA threads for each iteration of the parallel loops in the pseudocode, and using conditional execution in the kernel to determine which threads actually need to do work.
Such a solution will not be performant! (Consider the last outmost loop iteration of the upsweep phase, where only two threads would do work!).
A full credit solution will only launch one CUDA thread for each iteration of the innermost parallel loops.

**Test Harness:** By default, the test harness runs on a pseudo-randomly generated array that is the same every time the program is run, in order to aid in debugging.
You can pass the argument `-i random` to run on a random array - we will do this when grading.
We encourage you to come up with alternate inputs to your program to help you evaluate it.
You can also use the `-n <size>` option to change the length of the input array.

The argument `--thrust` will use the [Thrust Library's](http://thrust.github.io/) implementation of [exclusive scan](http://thrust.github.io/doc/group__prefixsums.html). Can you compete with Thrust?
