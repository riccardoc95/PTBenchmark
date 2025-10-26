Overview
============================

.. figure:: images/logo_color_light.png
    :align: center


Calculation of Persistent Homology for Pixel Data
*************************************************
PixHomology is an open-source software for image processing and analysis focused on persistent homology computation. It provides a set of tools and algorithms to explore the topological features of 2D images, enabling users to extract meaningful information about the underlying structures.

Features
********
- In its initial release, the software enables the computation of 0-dimensional homology on a 2D image.
- The construction of simplicial complexes involves establishing connections between each pixel and its 8 neighboring pixels.

Installation
************
PixHomology is available on PyPI and can be installed using pip. To install the package, run the following command:

**Optional** (create new environment):

.. code-block:: bash

   conda create -n pixh python=3.11
   conda activate pixh

**Install PixHomology**:

.. code-block:: python

   pip install pixhomology


**Building from Source**:

To build the source code, ensure you have CMake installed on your system. You can download CMake from `cmake.org <cmake.org>`_ or install it using your package manager.
To install PixHomology, follow these steps:

.. code-block:: python

    git clone https://github.com/riccardoc95/PixHomology.git
    cd PixHomology
    git submodule init
    git submodule update
    pip install .

For detailed installation instructions, please refer to the `Installation Guide <installation.html>`_.

Usage
*****
Here is a basic example of using PixHomology in Python:

.. code-block:: python

    import pixhomology as px
    image = np.random.rand(10,10)
    dgm = px.computePH(image)
    ...

For more examples and detailed usage instructions, check out the `Tutorials and examples <usage.html>`_.

To test the performance of PixHomology follow the instructions from `this repository <https://github.com/riccardoc95/TS2Benchmark>`_.
