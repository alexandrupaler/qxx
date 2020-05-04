from training import depth_range, gdv_name, benchmark

import getopt, sys
from statistics import mean
import random


# this is meant to be called from WRS
def main_wrs(argumentList):
    unixOptions = "w:d:b:c:e:m:r:s:"
    gnuOptions = ["max_breadth=", "max_depth=", "attr_b=", "attr_c=", "edge_cost=", "movement_factor=",
                  "use_random_circuit=", "seed="]

    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    argumentsDict = dict(arguments)

    # if this is true, a single circuit randomly chosen will be used for evaluation
    use_random_circuit = bool(getValue(argumentsDict, '-r', '--use_random_circuit', False))

    # the random generator seed
    seed = int(getValue(argumentsDict, '-s', '--seed', 1234567890))

    # 1 to 55 increment of 1
    max_breadth = int(getValue(argumentsDict, '-w', '--max_breadth', 1))

    # 2 to 55 increment of 1
    max_depth = int(getValue(argumentsDict, '-d', '--max_depth', 2))

    # -50 to 50 increment of 0.1
    attr_b = float(getValue(argumentsDict, '-b', '--attr_b', -50))

    # -1 to 1 increment of 0.1
    attr_c = float(getValue(argumentsDict, '-c', '--attr_c', -1))

    # 0.1 to 1 increment of 0.1
    edge_cost = float(getValue(argumentsDict, '-e', '--edge_cost', 0.1))

    # 1 to 55 increment of 1
    movement_factor = int(getValue(argumentsDict, '-m', '--movement_factor', 1))

    print("-----")
    print("Evaluating for max_breadth={}, max_depth={}, attr_b={}, attr_c={}, edge_cost={}, movement_factor={}".format(
        max_breadth,
        max_depth,
        attr_b,
        attr_c,
        edge_cost,
        movement_factor
    ))

    # compute target value (score) here !!!
    parameters = {
        "max_depth": max_depth,
        "max_children": max_breadth,

        "att_b": attr_b,
        "att_c": attr_c,

        "cx": edge_cost,
        "div_dist": movement_factor,

        # UNUSED
        "opt_att": True,
        "opt_max_t_min": False,
        "qubit_increase_factor": 3,
        "option_skip_cx": False,
        "penalty_skip_cx": 20,
        "opt_div_by_act": False,
    }

    scores = []

    if use_random_circuit:
        # maintain reproducibility
        random.seed(seed)
        trail = random.randint(0, 10)
        depthindex = random.randint(0, len(depth_range[gdv_name]))
        depth = depth_range[gdv_name][depthindex]
        # return optimal_depth, depth_result, execution_time, init_time, nr_t1, nr_t2
        res = benchmark(depth, trail, parameters)
        # since we know optimal_depth and depth_result I guess it would make sense to optimize their square diff.
        score = (res[0] - res[1]) ** 2
        scores.append(score)
        print(res)

    for trail in range(10):
        for depth in depth_range[gdv_name]:
            # return optimal_depth, depth_result, execution_time, init_time, nr_t1, nr_t2
            res = benchmark(depth, trail, parameters)
            # since we know optimal_depth and depth_result I guess it would make sense to optimize their square diff.
            score = (res[0] - res[1]) ** 2
            scores.append(score)
            print(res)

    # optimizing avg((optimal_depth-depth_result)**2)
    target_value = mean(scores)

    print("Target value is {}".format(target_value))
    print("______")

    # This is to return the value to the caller
    sys.stdout.write(str(target_value))
    sys.stdout.flush()
    sys.exit(0)


def getValue(dictionary, shortKey, longKey, default):
    return dictionary.get(shortKey, dictionary.get(longKey, default))


if __name__ == "__main__":
    main_wrs(sys.argv[1:])
