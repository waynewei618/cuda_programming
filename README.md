# GPGPU Programming with CUDA
This repository contains all code from the YouTube series "CUDA Crash Course (v3)" by CoffeeBeforeArch.

## Contact

Suggestions for specific content can be sent to: CoffeeBeforeArch@gmail.com

## Environment 

Operating System: Ubuntu 20.04

Text Editor: VIM

GPU: NVIDIA GTX 2060

CUDA version: 11

## Content

0. Introduction
1. Vector Addition
2. Matrix Multiplication
3. Sum Reduction
4. Histogram
5. Convolution

## Development Environment (VS Code & Docker)

This repository includes pre-configured settings for developing within the `ubuntu_dev` Docker container from a host VS Code editor:

- **Clangd Autocomplete**: Configured via [.clangd](.clangd) and [clangd-docker.sh](clangd-docker.sh) to forward syntax analysis and jumping to the container's clangd instance, enabling full CUDA language support in the host editor.
- **VS Code Tasks**: 
  - `CUDA Build (Docker)`: Press `Ctrl+Shift+B` to compile the currently active `.cu` file into a `<filename>.out` binary in the container.
  - `CUDA Run (Docker)`: Run the compiled `.out` binary in the container.
