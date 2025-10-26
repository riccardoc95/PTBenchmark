![Light](https://github.com/riccardoc95/PixHomology/blob/main/docs/images/logo_color_light.png#gh-light-mode-only)
![Dark](https://github.com/riccardoc95/PixHomology/blob/main/docs/images/logo_color_dark.png#gh-dark-mode-only)

# Calculation of Persistent Homology for Pixel Data
[![version](https://img.shields.io/badge/version-0.0.1-version.svg)](https://github.com/riccardoc95/PixHomology/releases/tag/alpha_v0.0.1)
[![version](https://img.shields.io/badge/python->8.0.0-version.svg)](https://www.python.org/)
[![version](https://img.shields.io/badge/gcc-15.0.0-version.svg)](https://gcc.gnu.org/)

PixHomology is an open-source software for image processing and analysis focused on persistent homology computation. It provides a set of tools and algorithms to explore the topological features of 2D images, enabling users to extract meaningful information about the underlying structures.

## Features

- The software enables the computation of 0-dimensional and 1-dimensional homology on a 2D image.
- The construction of simplicial complexes involves establishing connections between each pixel and its 8 neighboring pixels.

## Installation

**Optional** (create new environment):
```bash
conda create -n pixh python=3.11
conda activate pixh
```

### Python package
```bash
pip install pixhomology
```

### Building
> To build the source code, ensure you have CMake installed on your system. You can download CMake from [cmake.org](cmake.org) or install it using your package manager.
To install PixHomology, follow these steps:

```bash
git clone https://github.com/riccardoc95/PixHomology.git
cd PixHomology
git submodule init
git submodule update
pip install .
```

For detailed installation instructions, please refer to the [Installation Guide](docs/installation.rst).

## Usage

- Here is a basic example of using PixHomology in Python:

```python
import pixhomology as px
image = np.random.rand(10,10)
dgm = px.computePH(image, maxdim=0)
...
```

- Return also the positions (xb, yb, xd, yd) in pixels:

```python
import pixhomology as px
image = np.random.rand(10,10)
dgm, dgm_idx = px.computePH(image, return_index=True)
...
```
For more examples and detailed usage instructions, check out the [Usage Guide](docs/usage.rst).

- To test the performance of PixHomology follow the instructions from [this repository](https://github.com/riccardoc95/TS2Benchmark).

## Documentation

Comprehensive [documentation](https://riccardoc95.github.io/PixHomology/) includes installation guides, usage examples, and API references is available.

## Contributing

We welcome contributions from the community. If you want to contribute to PixHomology, please check the [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

PixHomology is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.
