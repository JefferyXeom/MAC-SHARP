# MAC-SHARP

Source code of the paper: (Still in development)

We firstly consider the parallel architecture for better performace 



## Requirements

### Hardware
Tested on modern INTEL CPU (with AVX instruction sets) 

### System
Tested on Windows 11 and Ubuntu 24.04

### C++ Compiler
Tested with GCC which support C++17 or later

### C++ libraries
PCL (Point Cloud Library) 1.12.0 or later, with AVX compilation flag enabled
OpenBLAS


## Installation

### 1. Install the required libraries first

#### 1.1 pcl

Go to the [PCL installation guide](https://pointclouds.org/downloads/) and follow the instructions for your platform.



``` bash
git clone https://github.com/JefferyXeom/MAC_SHARP.git
cd MAC_SHARP




```

## Tips in code

### 1. cpp standard

Our code is written in C++17 standard. If you want to compile it, please make sure your compiler supports C++17 or later.
However, we only use some of the C++17 features, so you can also compile it with C++14 or lower standard by removing the following keywords:

1. Inline variable: Those only used for cross file reference. There is replacement in lower standards.
2. nodiscard: The one used for functions that return a value that should not be ignored.


### 2. ??

All section bounded with
``` cpp
//---------------------------- Evaluation part ----------------------------
...
// -------------------------------------------------------------------------
```
is for evaluation only. They are irrelevant to the algorithm itself and use ground truth to compute the error.
You can remove them if you want to use the code in real applications.




## Reference
* [Zhang et al., 3D Registration with Maximal Cliques (CVPR 2023)](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
* [Zhang et al., MAC++: Going Further with Maximal Cliques for 3D Registration (3D Vision 2025)](https://github.com/zhangxy0517/MAC-PLUS-PLUS)
* [Zhang et al., FastMAC: Stochastic Spectral Sampling of Correspondence Graph (CVPR 2024)](https://github.com/Forrest-110/FastMAC)