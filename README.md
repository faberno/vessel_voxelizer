
<a id="readme-top"></a>
<!--
README Template from: https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="https://github.com/faberno/vessel_voxelizer/blob/main/files/logo.svg" alt="Logo" width="1000" height="100">
  </a>

  <h3 align="center">Vessel Voxelizer</h3>

  <p align="center">
    GPU accelerated (fuzzy) voxelization of vascular structures
    <br /><br />
    <a href="example/example.ipynb">Demo</a>
    ·
    <a href="https://github.com/faberno/vessel_voxelizer/issues">Report Bug / Request Feature</a>
    ·
    <a href="https://github.com/faberno/vessel_voxelizer/issues">Documentation</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#documentation">Documentation</a>
      <ul>
        <li><a href="#how it works">How it works</a></li>
        <li><a href="#API">Installation</a></li>
        <li><a href="#API">Example</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
**vessel-voxelizer** is a CUDA-accelerated tool designed to convert vascular structures, defined by line segments with associated radii, into fuzzy voxel volumes, where
each voxel's value represents the fraction of its volume occupied by the vessels. This fuzzy representation is essential for simulations where the volume fraction plays 
a critical role in assigning the correct parameters to each voxel, ensuring precise modeling of e.g. physical processes. 
The project leverages CUDA for high performance and includes Python bindings for seamless integration into existing simulation workflows.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- a CUDA-capable GPU

### Installation
1) clone the repository and update the nanobind submodule
```bash
git clone https://github.com/faberno/vessel_voxelizer.git
git submodule update --init --recursive
```
2) install cupy based on your CUDA version
```bash
# for CUDA 11.x
pip install cupy-cuda11x

# for CUDA 12.x
pip install cupy-cuda12x
```
3) compile and install the library
```bash
pip install .
```

<!-- USAGE EXAMPLES -->
## Documentation

### How it works

<div align="center">
  <a>
    <img src="files/howitworks.svg" alt="how_it_works" height="300">
  </a>
</div>


### API

### Example
For a full example, take a look at the following [notebook](example/example.ipynb).

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


