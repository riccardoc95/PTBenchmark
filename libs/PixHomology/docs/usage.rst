Tutorials and examples
============================


Basic Usage
----------------------------

1. **Importing the Library**

   First, import the necessary modules from PixHomology:

   .. code-block:: python

      import pixhomology as px

2. **Loading Image Data**

   Load an image using your preferred method. Here, we'll use `PIL` (Pillow) to load an image:

   .. code-block:: python

      from PIL import Image
      import numpy as np

      image_path = 'path_to_your_image.png'
      image = Image.open(image_path)
      image_data = np.array(image)

3. **Computing Homology**

   Compute the homology features from the image data:

   .. code-block:: python

      dgm = px.computePH(image_data)
      dgm, indexes = px.computePH(image_data, return_indexes=True)

4. **Visualizing Results**

    1. Plotting Persistence Diagrams
    To visualize the persistence diagram (PD) computed from image data:

    .. code-block:: python

        px.plotDGM(dgm)

    2.	Plotting Annotated Images
    To visualize the image with annotated birth and death points:

    .. code-block:: python

        px.plotIMG(image_data, indexes)

Advanced Usage
----------------------------

COMING SOON!
