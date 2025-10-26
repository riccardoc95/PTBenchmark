#ifndef PIXHOM_HPP
#define PIXHOM_HPP

// Define a struct to store the result
struct Result {
    std::vector <double> data;
    std::vector <int> posix;
    std::vector <int> edges;
    std::vector <int> labels;
    int length;
};

// Define a struct to store information about u_points
struct UPoint {
    int c_point;
    int u_point;
};

// PixHomology core functions
Result computePH(double* inputArr, int numR, int numC);
void freemem();

#endif // PIXHOM_HPP
