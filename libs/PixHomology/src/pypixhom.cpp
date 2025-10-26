#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pixhom.hpp"
#include "visual.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void change_sign(double* arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        arr[i] = -arr[i];
    }
}

py::object py_plot_dgm(py::array_t<double> input_array) {
    // Check the input dimensions
    if (input_array.ndim() != 2 || input_array.shape(1) != 2) {
        throw std::runtime_error("Input should be a 2D numpy array with shape (N, 2)");
    }

    // Extract data from the input array
    auto buf = input_array.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t num_points = buf.shape[0];

    std::vector<double> x(num_points), y(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        x[i] = ptr[i * 2];
        y[i] = ptr[i * 2 + 1];
    }

    // Call the scatterPlot function
    try {
        plotDGM(x, y);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in py_plot_dgm: ") + e.what());
    }

    return py::none();
}



py::object py_plot_img(py::array_t<double> input_array, py::array_t<double> indexes) {
    // Check the input dimensions
    if (indexes.ndim() != 2 || indexes.shape(1) != 4) {
        throw std::runtime_error("Input should be a 2D numpy array with shape (N, 2)");
    }

    // Ensure the input array is contiguous and 1D/2D
    py::buffer_info buf_info = input_array.request();
    int numRows;
    int numCols;
    if (buf_info.ndim == 2) {
        if (buf_info.shape[0] == 1 & buf_info.shape[1] != 1){
            numRows = buf_info.shape[1];
            numCols = 1;
        } else{
            numRows = buf_info.shape[0];
            numCols = buf_info.shape[1];
        }
    } else if (buf_info.ndim == 1){
        numRows = buf_info.shape[0];
        numCols = 1;
    }else{
        throw std::runtime_error("Input should be a 1D/2D NumPy array");
    }

    // Extract data from the input array
    auto buf_img = input_array.request();
    double* ptr_img = static_cast<double*>(buf_img.ptr);

    std::vector<double> img(numRows * numCols);
    for (size_t i = 0; i < numRows * numCols; ++i) {
        img[i] = ptr_img[i];
    }

    auto buf_idx = indexes.request();
    double* ptr_idx = static_cast<double*>(buf_idx.ptr);
    size_t num_points = buf_idx.shape[0];

    std::vector<double> x_birth(num_points), y_birth(num_points), x_death(num_points), y_death(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        x_birth[i] = ptr_idx[i * 4];
        y_birth[i] = ptr_idx[i * 4 + 1];
        x_death[i] = ptr_idx[i * 4 + 2];
        y_death[i] = ptr_idx[i * 4 + 3];
    }

    // Call the scatterPlot function
    try {
        plotIMG(img, numRows, numCols, x_birth, y_birth, x_death, y_death);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in py_plot_img: ") + e.what());
    }

    return py::none();
}


py::object py_computePH(py::array_t<double> input_array, bool return_index = false, int maxdim = 0) {
    // Ensure the input array is contiguous and 1D/2D
    py::buffer_info buf_info = input_array.request();
    int numRows;
    int numCols;
    if (buf_info.ndim == 2) {
        if (buf_info.shape[0] == 1 & buf_info.shape[1] != 1){
            numRows = buf_info.shape[1];
            numCols = 1;
        } else{
            numRows = buf_info.shape[0];
            numCols = buf_info.shape[1];
        }
    } else if (buf_info.ndim == 1){
        numRows = buf_info.shape[0];
        numCols = 1;
    }else{
        throw std::runtime_error("Input should be a 1D/2D NumPy array");
    }

    // Call the computePH function (dim = 0)
    Result res;
    try {
        res = computePH((double*)buf_info.ptr, numRows, numCols);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in computePH (dim = 0): ") + e.what());
    }

    // Create numpy arrays to return
    py::array_t<double> data_array_0({res.length / 2, 2});
    auto data_buf = data_array_0.request();
    std::copy(res.data.begin(), res.data.end(), (double *)data_buf.ptr);

    py::array_t<int> posix_array_0({res.length / 2, 4});
    auto posix_buf = posix_array_0.request();
    for (size_t i = 0; i < (res.posix.size() * 2); i += 4) {
        ((int*)posix_buf.ptr)[i] = res.posix[i / 2] % numCols;
        ((int*)posix_buf.ptr)[i + 1] = res.posix[i / 2] / numCols;
        ((int*)posix_buf.ptr)[i + 2] = res.posix[(i / 2) + 1] % numCols;
        ((int*)posix_buf.ptr)[i + 3] = res.posix[(i / 2) + 1] / numCols;
    }

    py::array_t<double> edges_0({numRows, numCols});
    auto edges_buf = edges_0.request();
    std::copy(res.edges.begin(), res.edges.end(), (double *)edges_buf.ptr);

    py::array_t<double> labels_0({numRows, numCols});
    auto labels_buf = labels_0.request();
    std::copy(res.labels.begin(), res.labels.end(), (double *)labels_buf.ptr);
    freemem();

    return py::make_tuple(data_array_0, posix_array_0, edges_0, labels_0);

}

PYBIND11_MODULE(pixhomology, m) {
    m.doc() = "PixHomology is an open-source software for image processing and analysis focused on persistent homology computation. It provides a set of tools and algorithms to explore the topological features of 1D series and 2D images, enabling users to extract meaningful information about the underlying structures."; // Optional module docstring

    m.def("computePH", &py_computePH,
      "A function that computes Persistent Homology (PH) on 1D series or 2D image data.\n\n"
      "Parameters:\n"
      "  input_array (numpy.ndarray): A 1D/2D array representing the input image data on which Persistent Homology is to be computed.\n"
      "  return_index (bool, optional): A flag indicating whether to return indices of the features. Defaults to False.\n"
      "  maxdim (int, optional): The maximum dimension of homology to compute. Defaults to 0.\n\n"
      "Returns:\n"
      "  numpy.ndarray or tuple:\n"
      "    - dgms (numpy.ndarray or list of numpy.ndarray): If maxdim=0, an (N, 2) array with birth and death values of dimension zero. If maxdim=1, a list of arrays for dimension 0 and dimension 1.\n"
      "    - indexes (numpy.ndarray or list of numpy.ndarray): If return_index=True. If maxdim=0, an (N, 4) array with x_birth, y_birth, x_death, y_death coordinates in pixels of dimension zero PH. If maxdim=1, a list of arrays for dimension 0 and dimension 1.",
      py::arg("input_array"),
      py::arg("return_index") = false,
      py::arg("maxdim") = 0);


    m.def("plotDGM", &py_plot_dgm,
          "A function that takes a 2D NumPy array (DGM) with dimensions (N, 2) as input and plots a persistence diagram. The first column of the input array corresponds to the x values (birth times), and the second column corresponds to the y values (death times).\n\n"
          "Parameters:\n"
          "  input_array (numpy.ndarray): A 2D array representing the persistence diagram (DGM) to be plotted.\n\n"
          "Returns:\n"
          "  None: The function plots the persistence diagram using matplotlib.");

    m.def("plotIMG", &py_plot_img,
          "A function that takes 2D NumPy arrays, image and DGM indexes, as input and plots image with birth and death points for each component.\n\n"
          "Parameters:\n"
          "  image (numpy.ndarray): A 2D array representing the image.\n\n"
          "  indexes (numpy.ndarray): A 2D array with dim (N, 4) representing the DGM indexes.\n\n"
          "Returns:\n"
          "  None: The function plots the image with the persistence information.");
    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
