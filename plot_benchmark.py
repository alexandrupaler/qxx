import csv
from ast import literal_eval
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

gdv_name = "TFL"

# Large compare plot
fig_large, ax_large = plt.subplots()
large_i = 0

def plot_experiment_results(benchmark_name):
    global large_i
    large_i += 1

    print(large_i, benchmark_name)
    folder_name = os.path.dirname(benchmark_name)


    # for future qiskit experiment
    dataset = list()

    depth_ratio = {
        "cirq": [0] * 9,
        "qiskit": [0] * 9,
        "tket": [0] * 9,
        "jku": [0] * 9,
        "k7m": [0] * 9
    }
    optimal_depth = [0] * 9

    """
        Start plotting
    """
    fig, ax = plt.subplots()

    # with open("_private_data/BNTF/{}".format(benchmark_name), 'r') as csvfile:
    with open(benchmark_name, 'r') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            data = list()
            for i in range(len(row)):
                if i is not 1:
                    data.append(literal_eval(row[i]))
                else:
                    data.append(row[i])

            dataset.append(data)
    # csvfile.close()


    """
    Generate data for k7m
    """
    # for tool in ["cirq", "qiskit", "tket", "jku", "k7m"]:
    for tool in ["k7m"]:
        for i in range(9):
            depth = 5 * (i + 1)
            optimal_depth[i] = depth
            count_data = 0
            for data in dataset:
                if data[1] == tool and data[2] == depth:
                    count_data += 1
                    depth_ratio[tool][i] += data[3] / data[2]

                    ax.plot(data[2], data[3] / data[2], 'o', color="lightgreen")
                    # with open("_private_data/BNTF/{}_{}.csv".format(gdv_name, tool), 'a') as csvfile:
                    #     csv.writer(csvfile).writerow([data[2], data[3] / data[2]])
                    # csvfile.close()
            depth_ratio[tool][i] /= count_data



        # for i in range(9):
        #     depth = 5 * (i + 1)
        #     with open("_private_data/BNTF/{}_{}.csv".format(gdv_name, tool), 'a') as csvfile:
        #         csv.writer(csvfile).writerow([depth, depth_ratio[tool][i]])
        #     csvfile.close()
        """
        Save AVG
        """
        # with open("{}_{}.avg".format(benchmark_name, tool), 'w') as csvfile:
        with open("{}/{}_{}.avg".format(folder_name,large_i ,tool), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(9):
                depth = 5 * (i + 1)
                writer.writerow([depth, depth_ratio[tool][i]])
        csvfile.close()

    """
    Load other files
    """

    # other_tools = ["cirq", "qiskit", "tket", "jku"]
    other_tools = []

    counts = [0] * 9
    for tool in other_tools:
        counts = [0] * 9
        with open("_private_data/BNTF/{}_{}.csv".format(gdv_name, tool), 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                i = (int(row[0]) - 5)//5
                depth_ratio[tool][i] += float(row[1])
                counts[i] += 1
        for i in range(9):
            depth_ratio[tool][i] /= counts[i]

    """
    Generate plots
    """

    for tool in other_tools + ["k7m"]:
        ax.plot(optimal_depth, depth_ratio[tool], label=tool)

        # if large_i == 4:
        #     ax_large.plot(optimal_depth, depth_ratio[tool], "x", label=large_i)
        # else:
        ax_large.plot(optimal_depth, depth_ratio[tool], label=large_i)

    ax.set(xlabel='Optimal Depth', ylabel='Depth Ratio')
    ax.set_ylim(0, 6)

    if len(other_tools) > 0:
        ax.legend()

    # fig.savefig('_private_data/BNTF/{}.png'.format(gdv_name), dpi=300)
    png_name = os.path.basename(benchmark_name)
    folder_name = os.path.dirname(benchmark_name)
    # fig.savefig('{}/{}{}.png'.format(folder_name,large_i,png_name), dpi=150)
    fig.savefig('{}/{}.png'.format(folder_name,large_i), dpi=150)


"""
    MAIN
"""
"""
    Configure existing benchmark names
"""
# benchmark_name = "TFL_Aspen-4_{'max_depth': 16, 'max_children': 3, 'option_max_then_min': False, 'qubit_increase_factor': 3, 'option_skipped_cnots': False, 'penalty_skipped_cnot': 200, 'option_divide_by_activated': False, 'option_attenuate': False}"

results_folder = "_private_data/BNTF/"
for file in os.listdir(results_folder):
    if file.startswith("_") and file.endswith(".csv"):
        bench_name = os.path.join(results_folder, file)
        plot_experiment_results(bench_name)

ax_large.set(xlabel='Optimal Depth', ylabel='Depth Ratio')
# ax_large.set_ylim(22, 27)
ax_large.legend()

fig_large.savefig('compare_all.png', dpi=150)