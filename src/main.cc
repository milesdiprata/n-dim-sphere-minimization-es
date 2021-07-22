#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "sphere/func_solver.h"

static constexpr const int kNumTests = 50;

int main(const int argc, const char* const argv[]) {
  auto csv_file = std::ofstream("fitnesses.csv", std::fstream::out);
  csv_file << "generation,fitness" << std::endl;

  sphere::FuncSolver<> solver(1.0);

  std::vector<double> fitnesses_avgs(sphere::FuncSolver<>::kNumGenerations,
                                     0.0);
  for (int i = 0; i < kNumTests; ++i) {
    auto fitnesses = solver.Start();
    for (std::size_t k = 0, size = fitnesses.size(); k < size; ++k) {
      fitnesses_avgs[k] += fitnesses[k];
    }
  }

  for (std::size_t k = 0, size = fitnesses_avgs.size(); k < size; ++k) {
    fitnesses_avgs[k] /= kNumTests;
    csv_file << k << "," << fitnesses_avgs[k] << std::endl;
  }

  csv_file.close();

  return 0;
}
