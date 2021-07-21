#include <iostream>

#include "es/adaptive_one_plus_one.h"

int main(const int argc, const char* const argv[]) {
  es::AdaptiveOnePlusOne<2>::Constraint constraint(-5.12, 5.12);
  es::AdaptiveOnePlusOne<2> solver(
      es::AdaptiveOnePlusOne<2>::Constraints({constraint, constraint}));
  auto individual = solver.Start();

  std::cout << individual << std::endl;
  return 0;
}
