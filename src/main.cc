#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>

#include "sphere/func_solver.h"

static constexpr const int kNumTests = 50;

int main(const int argc, const char* const argv[]) {
  auto csv_file = std::ofstream("fitnesses.csv", std::fstream::out);
  csv_file << "generation,fitness" << std::endl;

  sphere::FuncSolver<> solver(1.0);

  std::map<std::size_t, double> avg_fitnesses;
  for (int i = 0; i < kNumTests; ++i) {
    auto fitnesses = solver.Start();
    for (const auto& [generation, fitness] : fitnesses) {
      if (!avg_fitnesses.count(generation)) {
        avg_fitnesses[generation] = fitness;
      } else {
        avg_fitnesses[generation] += fitness;
      }
    }
  }

  for (auto& [generation, avg_fitness] : avg_fitnesses) {
    avg_fitness /= kNumTests;
    csv_file << generation << "," << avg_fitness << std::endl;
  }

  csv_file.close();

  return 0;
}
