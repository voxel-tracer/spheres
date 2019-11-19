#pragma once

#include <iostream>

struct options {
    int nx = 1200;
    int ny = 1200;
    int ns = 10;
    int dist = 100;
    int maxActivePaths = 1024 * 1024;
    int numBouncesPerIter = 4;
    char* colormap = "viridis.csv";
    char* input = NULL;
    bool verbose = false;
    bool binary = false;
};

int get_argi(int argc, char** argv, int idx) {
    if (idx >= argc) {
        std::cout << "invalid argument at index " << idx << std::endl;
        std::cout << "Run with - h to get full usage" << std::endl;
        exit(-1);
    }
    return strtol(argv[idx], NULL, 10);
}

bool parse_args(int argc, char** argv, options& opt) {
    int idx = 1;
    while (idx < argc) {
        char* arg = argv[idx];
        if (!strcmp(arg, "-i"))
            opt.input = argv[++idx];
        else if (!strcmp(arg, "-nx"))
            opt.nx = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-ny"))
            opt.ny = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-ns"))
            opt.ns = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-d"))
            opt.dist = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-mp"))
            opt.maxActivePaths = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-bi"))
            opt.numBouncesPerIter = get_argi(argc, argv, ++idx);
        else if (!strcmp(arg, "-c"))
            opt.colormap = argv[++idx];
        else if (!strcmp(arg, "-v"))
            opt.verbose = true;
        else if (!strcmp(arg, "-b"))
            opt.binary = true;
        else if (!strcmp(arg, "-h")) {
            std::cout << "usage: spheres -i <input file> [-b binary false] [-nx width 1200] [-ny height 1200] [-ns spp 10] [-d camera dist 100] [-mp max active paths 1M] [-bi numBouncesPerIter 4] [-c colormap viridis.csv] [-v verbose false]" << std::endl;
            return false;
        }
        else {
            std::cout << "invalid argument " << arg << std::endl;
            std::cout << "Run with -h to get full usage" << std::endl;
            return false;
        }

        idx++;
    }

    if (opt.input == NULL) {
        std::cout << "input file required" << std::endl;
        std::cout << "Run with -h to get full usage" << std::endl;
        return false;
    }

    return true;
}
