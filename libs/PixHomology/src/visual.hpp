#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

void plotDGM(const std::vector<double>& birth, const std::vector<double>& death);
void plotIMG(const std::vector<double>& img,
             const int numRows,
             const int numCols,
             const std::vector<double>& x_birth,
             const std::vector<double>& y_birth,
             const std::vector<double>& x_death,
             const std::vector<double>& y_death);

#endif // UTILS_HPP
