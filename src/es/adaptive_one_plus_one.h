#ifndef ES_ADAPTIVE_ONE_PLUS_ONE_H_
#define ES_ADAPTIVE_ONE_PLUS_ONE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace es {

template <std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<double, N>& array);

template <std::size_t N>
class AdaptiveOnePlusOne {
 public:
  struct Constraint {
    constexpr Constraint(
        const double lower = std::numeric_limits<double>::min(),
        const double upper = std::numeric_limits<double>::max())
        : lower(lower), upper(upper) {
      if (lower > upper) {
        throw std::invalid_argument("Lower bound is greater than upper bound!");
      }
    }

    constexpr ~Constraint() = default;

    double lower;
    double upper;
  };

  struct Individual {
    constexpr Individual() : fitness(std::numeric_limits<double>::max()) {}

    constexpr Individual(const Individual& individual)
        : object_params(individual.object_params),
          strategy_params(individual.strategy_params),
          fitness(individual.fitness) {}

    constexpr ~Individual() = default;

    friend std::ostream& operator<<(std::ostream& os,
                                    const Individual& individual) {
      return os << individual.object_params;
    }

    std::array<double, N> object_params;
    std::array<double, N> strategy_params;
    double fitness;
  };

  static constexpr const std::size_t kPopulationSize = 2;
  static constexpr const std::size_t kNumIndividualSelections = 1;
  static constexpr const std::size_t kNumParentSelections = 1;
  static constexpr const std::size_t kNumGenerations = 500;

  static constexpr const double kMutationMean = 0.0;
  static constexpr const double kInitialMutationStdDev = 0.02886751345;
  static constexpr const double kDefaultMutationStdDevFactor = 0.667489;
  static constexpr const double kPropSuccessfulMutationThreshold = 0.2;

  constexpr AdaptiveOnePlusOne(
      const std::optional<std::array<Constraint, N>>& constraints =
          std::nullopt,
      const double mutation_std_dev_factor = kDefaultMutationStdDevFactor)
      : constraints_(constraints),
        mutation_std_dev_factor_(mutation_std_dev_factor),
        mt_(std::random_device{}()) {}

  virtual constexpr ~AdaptiveOnePlusOne() = default;

  constexpr const std::optional<std::array<Constraint, N>>& constraints()
      const {
    return constraints_;
  }
  constexpr std::optional<std::array<Constraint, N>>& constraints() {
    return constraints_;
  }

  constexpr const double mutation_std_dev_factor() const {
    return mutation_std_dev_factor_;
  }

  constexpr double& mutation_std_dev_factor() {
    return mutation_std_dev_factor_;
  }

  const std::vector<double> Start() {
    std::vector<double> fitnesses;

    Individual first_individual = RandomIndividual();
    first_individual.fitness = Fitness(first_individual);
    fitnesses.push_back(first_individual.fitness);

    double mutation_std_dev = kInitialMutationStdDev;

    std::size_t num_successful_mutations = 0;
    std::size_t num_generations = 1;

    while (!Terminate(num_generations)) {
      std::array<double, N> random =
          RandomArray(kMutationMean, mutation_std_dev);

      Individual second_individual = first_individual;
      std::transform(second_individual.object_params.begin(),
                     second_individual.object_params.end(), random.begin(),
                     second_individual.object_params.begin(),
                     std::plus<double>());

      second_individual.fitness = Fitness(second_individual);
      if (second_individual.fitness < first_individual.fitness) {
        ++num_successful_mutations;
        first_individual = second_individual;
      }

      double prop_successful_mutations =
          (double)num_successful_mutations / num_generations;

      if (prop_successful_mutations < kPropSuccessfulMutationThreshold) {
        mutation_std_dev *= mutation_std_dev_factor_;
      } else if (prop_successful_mutations > kPropSuccessfulMutationThreshold) {
        mutation_std_dev /= mutation_std_dev_factor_;
      }

      fitnesses.push_back(first_individual.fitness);
      ++num_generations;
    }

    std::cout << "Solution: " << first_individual
              << " fitness: " << first_individual.fitness << std::endl;
    // return first_individual;
    return fitnesses;
  }

 protected:
  virtual constexpr const double Fitness(const Individual& individual) = 0;

 private:
  constexpr const bool Terminate(const std::size_t num_generations) const {
    return num_generations > kNumGenerations;
  }

  const Individual RandomIndividual() {
    std::uniform_real_distribution<> dis;
    Individual individual;

    for (std::size_t i = 0; i < N; ++i) {
      dis = constraints_.has_value() ? std::uniform_real_distribution<>(
                                           constraints_.value()[i].lower,
                                           constraints_.value()[i].upper)
                                     : std::uniform_real_distribution<>(
                                           std::numeric_limits<double>::min(),
                                           std::numeric_limits<double>::max());
      individual.object_params[i] = dis(mt_);
    }

    return individual;
  }

  const std::array<double, N> RandomArray(const double mean,
                                          const double std_dev) {
    std::normal_distribution<> dis(mean, std_dev);
    std::array<double, N> array;
    std::generate(array.begin(), array.end(),
                  [&]() -> const double { return dis(mt_); });

    return array;
  }

  std::optional<std::array<Constraint, N>> constraints_;
  double mutation_std_dev_factor_;
  std::mt19937_64 mt_;
};

template <std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<double, N>& array) {
  os << "[";
  std::string sep = "";

  for (const auto value : array) {
    os << sep << value;
    sep = ", ";
  }

  os << "]";

  return os;
}

}  // namespace es

#endif  // ES_ADAPTIVE_ONE_PLUS_ONE_H_
