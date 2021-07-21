#ifndef ES_ADAPTIVE_ONE_PLUS_ONE_H_
#define ES_ADAPTIVE_ONE_PLUS_ONE_H_

#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace es {

template <std::size_t N>
class AdaptiveOnePlusOne {
 public:
  struct Constraint {
    constexpr Constraint(
        const double lower = std::numeric_limits<double>::min(),
        const double upper = std::numeric_limits<double>::max())
        : lower(lower), upper(upper) {}

    constexpr ~Constraint() = default;

    double lower;
    double upper;
  };

  struct Individual {
    constexpr Individual()
        : object_params(N), strategy_params(N), constraints(constraints) {}

    constexpr Individual(const Individual& individual)
        : object_params(individual.object_params),
          strategy_params(individual.strategy_params),
          fitness(individual.fitness) {}

    constexpr Individual(Individual&& individual)
        : object_params(std::move(individual.object_params)),
          strategy_params(std::move(individual.strategy_params)),
          fitness(std::move(individual.fitness)) {}

    constexpr ~Individual() = default;

    std::vector<double> object_params;
    std::vector<double> strategy_params;
    std::optional<double> fitness;
  };

  using Constraints = std::optional<std::vector<Constraint>>;

  static constexpr const std::size_t kMaxNumGenerations = 150;

  constexpr AdaptiveOnePlusOne(const Constraints& constraints = std::nullopt)
      : constraints_(constraints), mt_(std::random_device{}()) {}

  virtual constexpr ~AdaptiveOnePlusOne() = default;

  constexpr const std::vector<double> Start() const {
    std::vector<double> solution = RandomSolution();

    return solution;
  }

 protected:
  virtual constexpr const double Fitness(
      const std::vector<double>& solution) = 0;

 private:
  constexpr const bool Terminate(const std::size_t num_generations) const {
    return num_generations > kMaxNumGenerations;
  }

  constexpr const std::vector<double> RandomSolution() {
    std::uniform_real_distribution<> dis;
    std::vector<double> solution(N);

    for (std::size_t i = 0; i < N; ++i) {
      if (constraints_.has_value()) {
        dis.a = constraints_.value()[i].lower;
        dis.b = constraints_.value()[i].upper;
      } else {
        dis.a = std::numeric_limits<double>::min();
        dis.b = std::numeric_limits<double>::max();
      }

      solution[i] = dis(mt_);
    }

    return solution;
  }

  Constraints constraints_;
  std::mt19937_64 mt_;
};

}  // namespace es

#endif
