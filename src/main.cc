#include <iostream>

#include "sphere/func_solver.h"

int main(const int argc, const char* const argv[]) {
  sphere::FuncSolver<> solver;
  auto individual = solver.Start();

  std::cout << "Solution: " << individual << " fitness: " << individual.fitness
            << std::endl;

  return 0;
}
