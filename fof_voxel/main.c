#include "allvars.h"
#include "proto.h"

#include <iostream>
#include <string>
#include <cstdlib>   // getenv
#include <chrono>

// -------------------------------------------------------------
// Pretty header with run configuration
// -------------------------------------------------------------
static void print_header(const char* prog, const std::string& in, const std::string& out) {
    std::cout << "\n=== FoF Halo Finder ===\n"
              << "Program: " << prog << "\n"
              << "Input :  " << (in == "-" ? "stdin (emu_input)" : in) << "\n"
              << "Output:  " << (out == "-" ? "stdout" : out) << "\n"
              << "LBOX=" << LBOX
              << "  LINKFACTOR=" << LINKFACTOR
              << "  MINGROUP=" << MINGROUP
              << "  BITS=" << BITS << "\n"
#ifdef _OPENMP
              << "OpenMP: enabled (OMP_NUM_THREADS="
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "default")
              << ")\n"
#else
              << "OpenMP: disabled\n"
#endif
              << "=======================\n" << std::flush;
}

// -------------------------------------------------------------
// main
// -------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_ascii|-> <output_ascii|->\n";
        return 1;
    }

    INPUTFILE = argv[1];   // "-" means stdin (handled in ReadTracers)
    GROUPFILE = argv[2];   // "-" could later be stdout if we extend WriteGroups

    print_header(argv[0], INPUTFILE, GROUPFILE);

    const auto t0 = std::chrono::steady_clock::now();

    try {
        ReadTracers();          // supports stdin if INPUTFILE == "-"
        Sort_part();
        Grid();

        Voxcel_FriendsOfFriends();
        FriendsOfFriends();

        WriteGroups();

        free(Tracer);
        FreeGrid();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 2;
    }
    catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
        return 2;
    }

    // optional: report total elapsed time
    const auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Total runtime: " << elapsed << " sec\n";

    return 0;
}
