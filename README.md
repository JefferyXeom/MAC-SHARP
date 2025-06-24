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






## Reference
* [Zhang et al., 3D Registration with Maximal Cliques (CVPR 2023)](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
* [Zhang et al., MAC++: Going Further with Maximal Cliques for 3D Registration (3D Vision 2025)](https://github.com/zhangxy0517/MAC-PLUS-PLUS)
* [Zhang et al., FastMAC: Stochastic Spectral Sampling of Correspondence Graph (CVPR 2024)](https://github.com/Forrest-110/FastMAC)