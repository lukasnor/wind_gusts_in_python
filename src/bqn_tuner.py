import bqn
from itertools import product, chain, combinations


def powerset(list):
    # powerset([1,2,3]) --> [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
    return map(lambda t: [i for i in t], chain.from_iterable(combinations(list, r) for r in range(len(list) + 1)))


# possibles values of the fixed params
horizons = [3, 6, 9, 12, 15, 18, 21, 24]
variables = ["u100", "v100", "t2m", "sp", "speed"]
variable_selections = [variables] # or =  list(powerset(variables))[1:]
aggregations = ["single", "single+std", "mean", "all"]
fixed_params_selections = [{"horizon": a, "variables": b, "aggregation": c} for a, b, c in
                      product(horizons, variable_selections, aggregations)]
print(variable_selections)
print(fixed_params_selections)
