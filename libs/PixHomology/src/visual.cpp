#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "visual.hpp"

namespace py = pybind11;

void plotDGM(const std::vector<double>& birth, const std::vector<double>& death) {
    // Import the matplotlib.pyplot module
    py::module_ plt = py::module_::import("matplotlib.pyplot");
    py::object scatter = plt.attr("scatter");
    py::object plot = plt.attr("plot");

    // Find the minimum and maximum of birth and death
    double min_val = std::min(*std::min_element(birth.begin(), birth.end()), *std::min_element(death.begin(), death.end()));
    double max_val = std::max(*std::max_element(birth.begin(), birth.end()), *std::max_element(death.begin(), death.end()));

    // Add the bisector line
    plot(py::make_tuple(min_val - 0.1, max_val + 0.1), py::make_tuple(min_val - 0.1, max_val + 0.1), "r--");

    // Plot the scatter plot
    scatter(death, birth);

    // Set the limits of the birth and death axes
    plt.attr("xlim")(py::make_tuple(min_val - 0.1, max_val + 0.1));
    plt.attr("ylim")(py::make_tuple(min_val - 0.1, max_val + 0.1));

    // Add labels and title
    plt.attr("xlabel")("Death");
    plt.attr("ylabel")("Birth");
    plt.attr("title")("DGM Plot");

    plt.attr("show")();

}

void plotIMG(const std::vector<double>& img,
             const int numRows,
             const int numCols,
             const std::vector<double>& x_birth,
             const std::vector<double>& y_birth,
             const std::vector<double>& x_death,
             const std::vector<double>& y_death){
    // Import the matplotlib.pyplot module
    py::module_ plt = py::module_::import("matplotlib.pyplot");
    py::object scatter = plt.attr("scatter");
    py::object imshow = plt.attr("imshow");

    // Plot the image
    py::array_t<double> image({numRows, numCols});
    auto data_buf = image.request();
    std::copy(img.begin(), img.end(), (double *)data_buf.ptr);
    imshow(image);

    // Plot the scatter plot
    scatter(x_birth, y_birth);
    scatter(x_death, y_death);

    // Add properties and title
    plt.attr("axis")("off");
    plt.attr("title")("Image with birth and death points");

    plt.attr("show")();


}
