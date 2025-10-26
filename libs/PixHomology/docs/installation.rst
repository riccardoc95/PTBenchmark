Installation
============================

Building from Source
----------------------------

To build and install PixHomology from source, follow these steps:

.. code-block:: bash

    git clone https://github.com/riccardoc95/PixHomology.git
    cd PixHomology
    git submodule init
    git submodule update
    pip install .

Installing via PyPI
-------------------

To install the latest stable release of PixHomology from PyPI, simply use:

.. code-block:: bash

    pip install pixhomology

Installing via Conda
----------------------------

If you prefer using Conda to manage your packages and environments, you can install PixHomology as follows:

.. code-block:: bash

    conda create -n pixhomology_env python=3.8
    conda activate pixhomology_env
    pip install pixhomology

Installing in a Virtual Environment
-----------------------------------

It is recommended to use a virtual environment to manage dependencies and isolate your project. Follow these steps to set up a virtual environment and install PixHomology:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pixhomology

Development Installation
------------------------

If you plan to contribute to the development of PixHomology, you can install it in editable mode, which allows you to make changes to the code and have them automatically reflected in your installed package:

.. code-block:: bash

    git clone https://github.com/riccardoc95/PixHomology.git
    cd PixHomology
    pip install -e .
