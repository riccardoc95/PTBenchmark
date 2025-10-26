#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>

#include "pixhom.hpp"

// Global Variables
int i = 0;
int j = 0;
int changed = 0;

int numRows = 0;
int numCols = 0;
int num_u_points = 0;
int num_dgm = 0;
int argmin = -1;
int argmax = -1;
int c_obj = 0;
int t_obj = 0;
int c_point = 0;
int t_point = 0;

double input = 0;
double min = -1;
double max = -1;
double localmax = -1;
double c_val = 0;
double t_val = 0;

double *inputArray = nullptr;
std::vector <int> mpatch;
std::vector <int> labels;
std::vector <int> dgm_pos;
std::vector <double> dgm;
std::vector <UPoint> uPoints;
Result res;


// Function to compare UPoints for qsort
bool compareUPoints(const UPoint& p1, const UPoint& p2) {
    double p1_u_val = inputArray[p1.u_point];
    double p2_u_val = inputArray[p2.u_point];
    double p1_c_val = inputArray[p1.c_point];
    double p2_c_val = inputArray[p2.c_point];

    if (p2_u_val > p1_u_val){
        return false;
    }else if (p2_u_val< p1_u_val){
        return true;
    }else{
        if (p2_c_val > p1_c_val){
            return false;
        }else if (p2_c_val < p1_c_val){
            return true;
        }else{
            return false;
        }
    }
}


// PixHomology core functions
Result computePH(double *inputArr, int numR, int numC) {
    // Set up
    inputArray = inputArr;
    numRows = numR;
    numCols = numC;

    // Calculate Argmin and Argmax
    if ((numRows * numCols) != 0) {
        min = inputArray[0];
        max = inputArray[0];
        argmin = 0;
        argmax = 0;

        for (i = 1; i < (numRows * numCols); i++) {
            input = inputArray[i];
            if (input <= min){
                min = input;
                argmin = i;
            }
            if (input >= max) {
                max = input;
                argmax = i;
            }
        }
    }

    // Set up mpatch array
    for (int i = 0; i < numRows * numCols; i++) {
        mpatch.push_back(i);
    }

    // First pass to find local maxima
    for (i = 0; i < numRows; i++) {
        for (j = 0; j < numCols; j++) {
            c_point = i * numCols + j;
            localmax = inputArray[c_point];
            t_point = 0;
            if ((i > 0) && (j > 0) && (i < numRows - 1) && (j < numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == 0) && (j > 0) && (j < numCols - 1)){
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i > 0) && (j == 0) && (i < numRows - 1)){
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == numRows - 1) && (j > 0) && (j < numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((j == numCols - 1) && (i > 0) && (i < numRows - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == 0) && (j == 0)){
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == numRows - 1) && (j == numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == numRows - 1) && (j == 0)){
                t_point = ((i - 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i - 1) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i) * numCols + (j + 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            } else if ((i == 0) && (j == numCols - 1)){
                t_point = ((i) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j - 1));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
                t_point = ((i + 1) * numCols + (j));
                input = inputArray[t_point];
                if (localmax < input) {
                    mpatch[c_point] = t_point;
                    localmax = input;
                }else if (localmax == input){
                    if (c_point < t_point){
                        mpatch[c_point] = t_point;
                    }else{
                        mpatch[c_point] = c_point;
                    }
                }
            }

        }
    }

    // Second pass to update the mpatch array
    while (1) {
        changed = 0;
        for (i = 0; i < numRows * numCols; i++) {
            c_obj = mpatch[mpatch[i]];
            if (mpatch[i] != c_obj) {
                mpatch[i] = c_obj;
                changed = 1;
            }
        }
        if (!changed) {
            break;
        }
    }

    // Set up UPoints
    num_u_points = 0;

    // Find uPoints
    for (i = 0; i < numRows; i++) {
        for (j = 0; j < numCols; j++) {
            c_point = i * numCols + j;
            t_point = 0;
            localmax = inputArray[c_point];
            if ((i > 0) && (j > 0) && (i < numRows - 1) && (j < numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == 0) && (j > 0) && (j < numCols - 1)){
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i > 0) && (j == 0) && (i < numRows - 1)){
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == numRows - 1) && (j > 0) && (j < numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((j == numCols - 1) && (i > 0) && (i < numRows - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == 0) && (j == 0)){
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == numRows - 1) && (j == numCols - 1)){
                t_point = ((i - 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == numRows - 1) && (j == 0)){
                t_point = ((i - 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i - 1) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i) * numCols + (j + 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            } else if ((i == 0) && (j == numCols - 1)){
                t_point = ((i) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j - 1));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
                t_point = ((i + 1) * numCols + (j));
                c_obj = mpatch[c_point];
                t_obj = mpatch[t_point];

                c_val = inputArray[c_point];
                t_val = inputArray[t_point];

                if (c_obj != t_obj && ((c_val > t_val) || ((c_val == t_val) && (c_point > t_point)))) {
                    uPoints.push_back(UPoint());

                    // Store information about u_point in the uPoints array
                    uPoints[num_u_points].c_point = c_point;
                    uPoints[num_u_points].u_point = t_point;

                    num_u_points = num_u_points + 1;

                }
            }

        }
    }

    // Sort uPoints in descending order
    sort(uPoints.begin(), uPoints.end(), compareUPoints);

    // Information about dgm
    num_dgm = 0;

    labels = mpatch;

    // Find dgm
    for (i = 0; i < num_u_points; i++) {
        c_point = uPoints[i].c_point;
        t_point = uPoints[i].u_point;

        c_obj = c_point;
        t_obj = t_point;

        while (c_obj != mpatch[c_obj]) {
            c_obj = mpatch[c_obj];
        }
        while (t_obj != mpatch[t_obj]) {
            t_obj = mpatch[t_obj];
        }

        if (c_obj != t_obj) {
            if (inputArray[c_obj] > inputArray[t_obj]) {
                mpatch[t_obj] = c_obj;
                if (fabs(inputArray[t_obj] - inputArray[t_point]) > 0) {
                    dgm.push_back(inputArray[t_obj]);
                    dgm.push_back(inputArray[t_point]);

                    dgm_pos.push_back(t_obj);
                    dgm_pos.push_back(t_point);

                    num_dgm = num_dgm + 2;
                }
            } else if (inputArray[c_obj] < inputArray[t_obj]) {
                mpatch[c_obj] = t_obj;
                if (fabs(inputArray[c_obj] - inputArray[t_point]) > 0) {
                    dgm.push_back(inputArray[c_obj]);
                    dgm.push_back(inputArray[t_point]);

                    dgm_pos.push_back(c_obj);
                    dgm_pos.push_back(t_point);

                    num_dgm = num_dgm + 2;
                }
            } else{
                if (c_obj > t_obj){
                    mpatch[t_obj] = c_obj;
                    if (fabs(inputArray[t_obj] - inputArray[t_point]) > 0) {
                        dgm.push_back(inputArray[t_obj]);
                        dgm.push_back(inputArray[t_point]);

                        dgm_pos.push_back(t_obj);
                        dgm_pos.push_back(t_point);

                        num_dgm = num_dgm + 2;
                    }
                }else{
                    mpatch[c_obj] = t_obj;
                    if (fabs(inputArray[c_obj] - inputArray[t_point]) > 0) {
                        dgm.push_back(inputArray[c_obj]);
                        dgm.push_back(inputArray[t_point]);

                        dgm_pos.push_back(c_obj);
                        dgm_pos.push_back(t_point);

                        num_dgm = num_dgm + 2;
                    }
                }

            }
        }
    }

    // Append the maximum and minimum values to dgm
    dgm.push_back(max);
    dgm.push_back(min);
    dgm_pos.push_back(argmax);
    dgm_pos.push_back(argmin);

    num_dgm = num_dgm + 2;

    res.data = dgm;
    res.posix = dgm_pos;
    res.length = num_dgm;
    res.edges = mpatch;
    res.labels = labels;

    return res;

}

void freemem(){
    i = 0;
    j = 0;
    changed = 0;

    numRows = 0;
    numCols = 0;
    num_u_points = 0;
    num_dgm = 0;
    argmin = -1;
    argmax = -1;
    c_obj = 0;
    t_obj = 0;
    c_point = 0;
    t_point = 0;

    input = 0;
    min = -1;
    max = -1;
    localmax = -1;
    c_val = 0;
    t_val = 0;

    inputArray = nullptr;

    res.data.clear();
    res.posix.clear();
    res.length = 0;

    mpatch.clear();
    dgm_pos.clear();
    dgm.clear();
    uPoints.clear();
}
