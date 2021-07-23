#ifndef SPHERE_FUNC_SOLVER_H_
#define SPHERE_FUNC_SOLVER_H_

#include <array>
#include <cstddef>

#include "es/adaptive_one_plus_one.h"

namespace sphere {

constexpr const std::size_t kDefaultSolutionSize = 10;

template <std::size_t N = kDefaultSolutionSize>
class FuncSolver : public es::AdaptiveOnePlusOne<N> {
 public:
  using Constraint = typename es::AdaptiveOnePlusOne<N>::Constraint;
  using Individual = typename es::AdaptiveOnePlusOne<N>::Individual;

  static constexpr const double kSolutionLowerLimit = -5.12;
  static constexpr const double kSolutionUpperLimit = 5.12;

  static constexpr const Constraint kConstraint =
      Constraint(kSolutionLowerLimit, kSolutionUpperLimit);

  constexpr FuncSolver()
      : es::AdaptiveOnePlusOne<N>(MakeConstraints(kConstraint)) {}

  constexpr FuncSolver(const double c_value)
      : es::AdaptiveOnePlusOne<N>(MakeConstraints(kConstraint), c_value) {}

  virtual constexpr ~FuncSolver() = default;

 protected:
  constexpr const double Fitness(const Individual& individual) final {
    double sum = 0;
    for (const auto& value : individual.object_params) {
      sum += (value * value);
    }

    return sum;
  }

 private:
  constexpr const std::array<Constraint, N> MakeConstraints(
      const Constraint& constraint) {
    std::array<Constraint, N> constraints;
    constraints.fill(constraint);
    return constraints;
  }
};

}  // namespace sphere

#endif  // SPHERE_FUNC_SOLVER_H_
