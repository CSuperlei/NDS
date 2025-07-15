# A sub-10ms neural dynamical system based on phase change memristors
<div align="center">
    <img src="https://github.com/CSuperlei/NDS/raw/main/Pic/NDS.png" alt="HIFT">
</div>

## Introduction
<div style="text-align: justify;">
The neural dynamical system (NDS) has emerged as a cutting-edge technology that synergizes the strengths of neural networks and the power of continuous-time dynamical systems for highly complex applications. However, NDS features adaptive stepsize numerical integration with embedded neural networks (ENNs), which naturally incur frequent yet numerous data movements and costly stepsize search iterations. To date, NDS runtime is limited to hundreds of ms with the classical multilayer perceptron ENN, severely hindering its practical adoption. Here we report a sub-10ms low-latency NDS inspired by two distinct characteristics of memristors, namely precisely controlled conductance drift and multilevel conductance. Using precisely controlled conductance drift (CCD) of PCM, an in-situ stepsize search is proposed that significantly reduces the latency and hardware cost for numerical integration. Furthermore, using multilevel conductance (MLC) up to eight levels, we fabricate highly uniform one-transistor-one-resistor (1T1R) phase-change memristor (PCM) arrays with efficient on-chip circuits for compute-in-memory and high weight storage density. We experimentally implement such an NDS based on a 40nm technology and evaluate its performance for highly complex 3D manifold rendering. Our NDS hardware demonstrates an end-to-end latency below 10ms with very high error tolerance precision of 10^{-5}, outperforming general-purpose CPU/GPU systems by 2.14x10^{3}x~1.08x10^{4}x. Compared with state-of-the-art NDS accelerators that perform the same tasks, our design delivers 13.86x~5089x faster speeds while consuming only 11.78x~24.79x less power. The exploitation of two distinct characteristics of memristors (MLC and CCD) is promising for pushing NDS into highly complex modeling applications under a reasonable runtime.
</div>


## Requirements
  * numpy>=1.20.0
  * scipy>=1.10.0
  * torch>=1.12.1
  * torchvision>=0.14.0
  * matplotlib>=3.7.0
  * open3d>=0.17.0
  * scikit-learn>=0.21.3
  * pandas>=2.0.2
  * TorchDiffEq

## Installation

### Anaconda
  bash Anaconda3-2023.09-0-Linux-x86_64.sh <br/>
### Pytorch 
   ```python
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

### Other Packages
 ```python
   pip install open3d   
   pip install PyQt5   
   pip install matplotlib
   pip install scipy
   pip install numpy
   pip install TorchDiffEq
   ```

## License
* This project is covered under the MIT License.
