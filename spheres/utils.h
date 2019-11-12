#pragma once

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

/**
* Reads csv file into table, exported as a vector of vector of doubles.
* @param inputFileName input file name (full path).
* @return data as vector of vector of doubles.
*
* code adapted from https://waterprogramming.wordpress.com/2017/08/20/reading-csv-files-in-c/
*/
std::vector<std::vector<float>> parse2DCsvFile(std::string inputFileName) {

    std::vector<std::vector<float> > data;
    std::ifstream inputFile(inputFileName);
    int l = 0;

    while (inputFile) {
        l++;
        std::string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            std::istringstream ss(s);
            std::vector<float> record;

            while (ss) {
                std::string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    std::cout << "NaN found in file " << inputFileName << " line " << l << std::endl;
                    e.what();
                }
            }

            data.push_back(record);
        }
    }

    if (!inputFile.eof()) {
        std::cerr << "Could not read file " << inputFileName << std::endl;
        exit(99);
    }

    return data;
}
