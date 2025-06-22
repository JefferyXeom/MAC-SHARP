//
// Created by Jeffery_Xeom on 2025/6/20.
//

#include <iostream>
#include <omp.h> // 包含 OpenMP 头文件
#include <vector> // 仅用于暂停程序观察输出

int main() {
    std::cout << "This program will test the OpenMP environment." << std::endl;
    std::cout << "The default number of threads available is: " << omp_get_max_threads() << std::endl;
    std::cout << "Starting parallel region..." << std::endl;

    // 这个 pragma 指示编译器让多个线程并行执行下面的代码块
#pragma omp parallel
    {
        // omp_get_thread_num() 获取当前线程的ID (从0开始)
        // omp_get_num_threads() 获取当前并行区域中实际运行的线程总数

        // #pragma omp critical 确保同一时间只有一个线程可以执行打印操作，防止输出混乱
#pragma omp critical
        {
            std::cout << "Hello from thread " << omp_get_thread_num()
                      << " of " << omp_get_num_threads() << std::endl;
        }
    }

    std::cout << "Parallel region finished." << std::endl;

    // 在Windows上，如果您直接双击运行，窗口可能会一闪而过
    // 这几行代码可以让程序暂停，直到您按回车键
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}