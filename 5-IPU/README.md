# Programming in the IPU accelerator of Grpahcore

In this tutorial we are going to use a software emulator for the IPU. I suggest that you work in a normal CPU cluster node. Taking 7 cores in an Iris node should be sufficient:
```bash
salloc --partition=interactive --qos=debug --time=2:00:00 --ntasks=1 --cpus-per-task=7
```
Download the development framework, Poplar SKD, from the [Grpahcore website](https://www.graphcore.ai/downloads):
```bash
curl --location --request GET 'https://downloads.graphcore.ai/direct?package=poplar-poplar_sdk_ubuntu_20_04_3.4.0_69d9d03fd8-3.4.0&file=poplar_sdk-ubuntu_20_04-3.4.0-69d9d03fd8.tar.gz' --output 'poplar_sdk-ubuntu_20_04-3.4.0-69d9d03fd8.tar.gz'
```
Then, extract the SDK:
```bash
tar xvf poplar_sdk_ubuntu_20_04_3-69d9d03fd8.tar.gz
```
To use the SDK enable the software environment:
```bash
source /path/to/poplar_sdk-ubuntu_20_04-3.4.0+1507-69d9d03fd8/enable
```

## The basic structure of IPU accelerated programs

The examples that we cover in this class are all located in the [official Graphcore example repository](https://github.com/graphcore/examples). Download the repository and navigate to:
```
tutorials/tutorials/poplar
```
This directory contains examples with Poplar, a low level library for programming the IPU. Unlike CUDA, there is no framework requiring a special compiler to interpret directives such as kernel calls. In Poplar you simply call the appropriate functions from a set of libraries, and at run time you link with the appropriate libraries. Any code that runs on the IPU can be compiled during run time, or it can be compiled beforehand in a machine code file that is simply loaded to the IPU at runtime.

## _Resources_

- [IPU Programmer's Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/): a description of the architecture of the IPU and a high level overview of the program API and ABI.
- Poplar and PopLibs: Detailed resource on the Poplar programming API.
    - [Poplar and PopLibs User Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/)
    - [Poplar and PopLibs API Reference](https://docs.graphcore.ai/projects/poplar-api/en/latest/)
- [A Dictionary of Graphcore Terminology](https://docs.graphcore.ai/projects/graphcore-glossary/en/latest/): Explanations of the terminology related to Graphcore infrastructure and links to resources.
