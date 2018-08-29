# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
---------------> do not modify this file for your submission <---------------

This file is not considered to be part of your submission.
It is only provided for you to benchmark your code.
The local_qasm_cpp_simulator that is used requires QISKit 0.43 or later.

"""

import numpy as np
import time
import copy
import qiskit
import sys, os, traceback

GLOBAL_TIMEOUT = 3600
ERROR_LIMIT = 1e-10
from qiskit import QuantumProgram
from qiskit.unroll import Unroller, DAGBackend
from qiskit._openquantumcompiler import dag2json
from multiprocessing import Pool
from qiskit.mapper._mappererror import MapperError
from qiskit.tools.qi.qi import state_fidelity

#paler
import json

skipVerif = True

def score(compiler_function=None, backend = 'local_qiskit_simulator'):
    """
    Scores a compiler function based on a selected set of circuits and layouts
    available in the two respective subfolders.
    The final scoring will be done on a similar set of circuits and layouts.


    Args:
        compiler_function (function): reference to user compiler function

    Returns:
        float : score, speed
    """
    # Load coupling maps
    maps_q5 = ["circle_rand_q5","ibmqx2_q5","linear_rand_q5","ibmqx4_q5","linear_reg_q5"]
    maps_q16 = ["ibmqx3_q16", "linear_rand_q16", "rect_rand_q16", "rect_def_q16", "ibmqx5_q16"]
    maps_q20 = ["circle_reg_q20", "linear_rand_q20", "rect_rand_q20", "rect_def_q20", "rect_reg_q20"]

    # Load circuits files
    #TODO: Was 10
    ex_nr = 1  # examples to add per qubit number. maximum is 10
    test_circuit_filenames = {}

    # The following two for loops are to test the circuits that failed at the Qiskit challenge
    for ii in range(5):
        test_circuit_filenames['circuits/qft_n16.qasm'] = load_coupling(maps_q16[ii%len(maps_q16)])["coupling_map"]

    for ii in range(5):
        test_circuit_filenames['circuits/qft_n16_excitations.qasm'] = load_coupling(maps_q16[ii%len(maps_q16)])["coupling_map"]

    for ii in range(ex_nr):
        test_circuit_filenames['circuits/random%d_n5_d5.qasm' % ii] = load_coupling(maps_q5[ii%len(maps_q5)])["coupling_map"]
    # for ii in range(ex_nr):
    #     test_circuit_filenames['circuits/random%d_n16_d16.qasm' % ii] = load_coupling(maps_q16[ii%len(maps_q16)])["coupling_map"]
    # for ii in range(ex_nr):
    #     test_circuit_filenames['circuits/random%d_n20_d20.qasm' % ii] = load_coupling(maps_q20[ii%len(maps_q20)])["coupling_map"]

    # Load example circuits and coupling maps

    test_circuits = {}
    for filename, cmap in test_circuit_filenames.items():
        with open(filename, 'r') as infile:
            qasm = infile.read()
            test_circuits[filename] = {"qasm": qasm, "coupling_map": cmap}
    res = evaluate(compiler_function, test_circuits, verbose=True, backend = backend)
    res_scores=[]
    for name in res:
        if (res[name]["optimizer_time"] > 0) and res[name]["coupling_correct_optimized"]:
            # only add the score if the QISKit reference compiler worked well
            if (res[name]["reference_time"] > 0) and res[name]["coupling_correct_reference"]:
                # both user and reference compiler gave the correct result without error
                res_scores.append([res[name]["cost_optimized"]/res[name]["cost_reference"],res[name]["optimizer_time"]/res[name]["reference_time"]])
        else:
            # the user compiler had an error or did not produce the right quantum state
            # this returns a value which is half as good as the reference
            res_scores.append([2,2])
    return (1./np.mean([ii[0] for ii in res_scores]), 1./np.mean([ii[1] for ii in res_scores]))


def evaluate(compiler_function=None, test_circuits=None, verbose=False, backend = 'local_qiskit_simulator'):
    """
    Evaluates the given complier_function with the circuits in test_circuits
    and compares the output circuit and quantum state with the original and
    a reference obtained with the qiskit compiler.

    Args:
        compiler_function (function): reference to user compiler function
        test_circuits (dict): named dict of circuits for which the compiler performance is evaluated

            test_circuits: {
                "name": {
                    "qasm": 'qasm_str',
                    "coupling_map": 'target_coupling_map
                }
            }

        verbose (bool): specifies if performance of basic QISKit unroler and mapper circuit is shown for each circuit
        backend (string): backend to use. For Windows Systems you should specify 'local_qasm_simulator' until
                         'local_qiskit_simulator' is available.



    Returns:
        dict
        {
            "name": circuit name
            {
                "optimizer_time": time taken by user compiler,
                "reference_time": reference time taken by qiskit circuit mapper/unroler (if verbose),
                "cost_original":  original circuit cost function value (if verbose),
                "cost_reference": reference circuit cost function value (if verbose),
                "cost_optimized": optimized circuit cost function value,
                "coupling_correct_original": (bool) does original circuit
                                                    satisfy the coupling map (if verbose),
                "coupling_correct_reference": (bool) does circuit produced
                                                    by the qiskit mapper/unroler
                                                    satisfy the coupling map (if verbose),
                "coupling_correct_optimized": (bool) does optimized circuit
                                                    satisfy the coupling map,
                "state_correct_optimized": (bool) does optimized circuit
                                                  return correct state
            }
        }
    """

    # Initial Setup
    basis_gates = 'u1,u2,u3,cx,id'  # or use "U,CX?"
    gate_costs = {'id': 0, 'u1': 0, 'measure': 0, 'reset': 0, 'barrier': 0,
                  'u2': 1, 'u3': 1, 'U': 1,
                  'cx': 10, 'CX': 10}
    # Results data structure
    results = {}

    fileJSONExists = False
    #paler json
    if os.path.isfile("run_once_results.json"):
        with open("run_once_results.json", "r") as f:
            fileJSONExists = True
            results = json.load(f)
            print("Paler: Loaded JSON")
    #end paler json


    # Load QASM files and extract DAG circuits
    for name, circuit in test_circuits.items():
        print("....name " + name)

        qp = QuantumProgram()
        qp.load_qasm_text(
            circuit["qasm"], name, basis_gates=basis_gates)
        circuit["dag_original"] = qasm_to_dag_circuit(circuit["qasm"], basis_gates=basis_gates)
        test_circuits[name] = circuit

        if not fileJSONExists:
            results[name] = {}  # build empty result dict to be filled later

    # Only return results if a valid compiler function is provided
    if compiler_function is not None:
        # step through all the test circuits using multiprocessing
        compile_jobs = [[name,circuit,0,compiler_function,gate_costs] for name, circuit in test_circuits.items()]
        with Pool(len(compile_jobs)) as job:
            res_values_opt = job.map(_compile_circuits, compile_jobs)
        # stash the results in the respective dicts
        print("..... [compiled optimised]")

        for job in range(len(compile_jobs)):
            name = res_values_opt[job].pop("name")
            test_circuits[name].update(res_values_opt[job].pop("circuit")) # remove the circuit from the results and store it
            #results[name] = res_values_opt[job]
            results[name].update(res_values_opt[job])
        # do the same for the reference compiler in qiskit if verbose == True
        # paler json
        if verbose and (not fileJSONExists):
            compile_jobs = [[name, circuit, 1, _qiskit_compiler, gate_costs] for name, circuit in
                            test_circuits.items()]
            with Pool(len(compile_jobs)) as job:
                res_values = job.map(_compile_circuits, compile_jobs)
            # also stash this but use update so we don't overwrite anything
            print("..... [compiled reference]")
            for job in range(len(compile_jobs)):
                name = res_values[job].pop("name")
                test_circuits[name].update(res_values[job].pop("circuit")) # remove the circuit from the results and store it
                results[name].update(res_values[job])


        # determine the final permutation of the qubits
        # this is done by analyzing the measurements on the qubits
        compile_jobs = [[name, circuit, verbose] for name, circuit in test_circuits.items()]
        with Pool(len(compile_jobs)) as job:
            res_values = job.map(_prep_sim, compile_jobs)

        for job in range(len(compile_jobs)):
            name = res_values[job].pop("name")
            test_circuits[name].update(res_values[job].pop("circuit")) # remove the circuit from the results and store it
            results[name].update(res_values[job])

        # Compose qobj for simulation
        config = {
            'data': ['quantum_state'],
        }

        # generate qobj for original circuit
        qobj_original = _compose_qobj("original", test_circuits,
                                                      backend=backend,
                                                      config=config,
                                                      basis_gates=basis_gates,
                                                      shots=1,
                                                      seed=None)

        # Compute original cost and check original coupling map
        if verbose and (not fileJSONExists):
            for circuit in qobj_original["circuits"]:
                name = circuit["name"]
                coupling_map = test_circuits[name].get("coupling_map", None)
                coupling_map_passes = True
                cost = 0
                for op in circuit["compiled_circuit"]["operations"]:
                    cost += gate_costs.get(op["name"])  # compute cost
                    if op["name"] in ["cx", "CX"] \
                            and coupling_map is not None:  # check coupling map
                        coupling_map_passes &= (
                            op["qubits"][0] in coupling_map)
                        if op["qubits"][0] in coupling_map:
                            coupling_map_passes &= (
                                op["qubits"][1] in coupling_map[op["qubits"][0]]
                            )

                results[name]["cost_original"] = cost
                results[name]["coupling_correct_original"] = coupling_map_passes

        # Run simulation
        if not skipVerif:
            time_start = time.process_time()
            res_original = qp.run(qobj_original, timeout=GLOBAL_TIMEOUT)
            print("..... [executed original]")
            results[name]["sim_time_orig"] = time.process_time() - time_start

        # Generate qobj for optimized circuit
        qobj_optimized = _compose_qobj("optimized", test_circuits,
                                                      backend=backend,
                                                      config=config,
                                                      basis_gates=basis_gates,
                                                      shots=1,
                                                      seed=None)

        # Compute compiled circuit cost and check coupling map
        for circuit in qobj_optimized["circuits"]:
            name = circuit["name"]
            coupling_map = test_circuits[name].get("coupling_map", None)
            coupling_map_passes = True
            cost = 0
            for op in circuit["compiled_circuit"]["operations"]:
                cost += gate_costs.get(op["name"])  # compute cost
                if op["name"] in ["cx", "CX"] \
                        and coupling_map is not None:  # check coupling map
                    coupling_map_passes &= (
                        op["qubits"][0] in coupling_map)
                    if op["qubits"][0] in coupling_map:
                        coupling_map_passes &= (
                            op["qubits"][1] in coupling_map[op["qubits"][0]]
                        )
            results[name]["cost_optimized"] = cost
            results[name]["coupling_correct_optimized"] = coupling_map_passes

        # Run simulation
        if not skipVerif:
            time_start = time.process_time()
            res_optimized = qp.run(qobj_optimized, timeout=GLOBAL_TIMEOUT)
            results[name]["sim_time_opti"] = time.process_time() - time_start
            print("..... [executed optimised]")

        # paler json
        if verbose and (not fileJSONExists):
            # Generate qobj for reference circuit optimized by qiskit compiler
            qobj_reference = _compose_qobj("reference", test_circuits,
                                                      backend=backend,
                                                      config=config,
                                                      basis_gates=basis_gates,
                                                      shots=1,
                                                      seed=None)

            # Compute reference cost and check reference coupling map
            for circuit in qobj_reference["circuits"]:
                name = circuit["name"]
                coupling_map = test_circuits[name].get("coupling_map", None)
                coupling_map_passes = True
                cost = 0
                for op in circuit["compiled_circuit"]["operations"]:
                    cost += gate_costs.get(op["name"])  # compute cost
                    if op["name"] in ["cx", "CX"] \
                            and coupling_map is not None:  # check coupling map
                        coupling_map_passes &= (
                            op["qubits"][0] in coupling_map)
                        if op["qubits"][0] in coupling_map:
                            coupling_map_passes &= (
                                op["qubits"][1] in coupling_map[op["qubits"][0]]
                            )
                results[name]["cost_reference"] = cost
                results[name]["coupling_correct_reference"] = coupling_map_passes

            # Skip simulation of reference State to speed things up!
            # time_start = time.process_time()
            # res_reference = qp.run(qobj_reference, timeout=GLOBAL_TIMEOUT)
            # results[name]["sim_time_ref"] = time.process_time() - time_start


        # Check output quantum state of optimized circuit is correct in comparison to original
        for name in results.keys():
            # handle errors here

            if skipVerif:
                results[name]["state_correct_optimized"] = True
                continue

            data_original = res_original.get_data(name)
            if test_circuits[name]["dag_optimized"]is not None:
                data_optimized = res_optimized.get_data(name)
                correct = _compare_outputs(data_original, data_optimized, test_circuits[name]["perm_optimized"])
                #paler transform np.bool_ to bool
                results[name]["state_correct_optimized"] = bool(correct)

                print(name, bool(correct))

                #skip verification
                # results[name]["state_correct_optimized"] = True
            else:
                results[name]["state_correct_optimized"] = False
            # Skip verification of the reference State to speed things up!
            # if verbose:
            #     if test_circuits[name]["dag_reference"] is not None:
            #         data_reference = res_reference.get_data(name)
            #         correct = _compare_outputs(data_original, data_reference, test_circuits[name]["perm_reference"])
            #         results[name]["state_correct_reference"] = correct
            #     else:
            #         results[name]["state_correct_reference"] = False

    # paler json
    with open("run_once_results.json", "w") as f:
        json.dump(results, f)
        print("Paler: Wrote JSON")
    # end paler json

    return results


def qasm_to_dag_circuit(qasm_string, basis_gates='u1,u2,u3,cx,id'):
    """
    Convert an OPENQASM text string to a DAGCircuit.

    Args:
        qasm_string (str): OPENQASM2.0 circuit string.
        basis_gates (str): QASM gates to unroll circuit to.

    Returns:
        A DAGCircuit object of the unrolled QASM circuit.
    """
    program_node_circuit = qiskit.qasm.Qasm(data=qasm_string).parse()
    dag_circuit = Unroller(program_node_circuit,
                           DAGBackend(basis_gates.split(","))).execute()
    return dag_circuit


def get_layout(qubits = 5):
    # load a random layout for a specifed number or qubits
    # only layouts for 5, 16 and 20 qubits are stored
    from os import path, getcwd
    from glob import glob
    from numpy.random import rand
    filenames = glob(path.join(getcwd(), 'layouts', '*_q' + str(qubits) + '*'))
    if len(filenames) > 0:
        file = filenames[round(rand()*len(filenames)-0.5)]
        return load_coupling(path.basename(file).split('.')[-2])["coupling_map"]
    else:
        return []


def load_coupling(name):
    """
    Loads the coupling map that is given as a json file from a subfolder named layouts

    Args:
        name (string): name of the coupling map that was used when saving
                        (corresponds to the filename without the extension .json)
    Returns:
        dict
        {
            "qubits" : the number of qubits in the coupling map
            "name": coupling map name
            "coupling_map" : actual coupling map as used by QISKit
            "position" : arrangement of the qubits in 2D if available
            "description": additional information on the coupling map
        }

    """
    import json
    with open("./layouts/"+name+".json", 'r') as infile:
        temp = json.load(infile)
        temp["coupling_map"] = {int(ii): kk for ii, kk in temp["coupling_map"].items()}
        return temp

def _compile_circuits(compile_args):
    name = compile_args[0]
    circuit = compile_args[1]
    comp_type = compile_args[2]
    compiler_function = compile_args[3]
    gate_costs = compile_args[4]
    res_values = {}
    res_values["name"] = name
    # initialize error
    circuit["error"] = {}
    if comp_type == 0:
        # Evaluate runtime of submitted compiler_function
        try:
            time_start = time.process_time()
            circuit["dag_optimized"] = compiler_function(
                circuit["dag_original"], coupling_map=circuit["coupling_map"],
                gate_costs=gate_costs)
            res_values["optimizer_time"] = time.process_time() - time_start
        except Exception as e:

            print(e)

            circuit["dag_optimized"] = None
            err = traceback.format_exc()
            circuit["error"]["optimized"] = [err, name, circuit["coupling_map"]]
            res_values["optimizer_time"] = -1
            print(
                "An error occurred in the user compiler while running " + name + " on " + str(
                    circuit["coupling_map"]) + '.')
            pass
    else:
        # Evaluate runtime of qiskit compiler
        try:
            time_start = time.process_time()
            circuit["dag_reference"] = _qiskit_compiler(
                circuit["dag_original"], coupling_map=circuit["coupling_map"],
                gate_costs=gate_costs)
            res_values["reference_time"] = time.process_time() - time_start
        except Exception as e:

            print(e)

            circuit["dag_reference"] = None
            err = traceback.format_exc()
            circuit["error"]["reference"] = [err, name, circuit["coupling_map"]]
            res_values["reference_time"] = -1
            print("An error occurred in the QISKit compiler while running " + name + " on " + str(
                circuit["coupling_map"]) + '.')
            pass

    # Store the compiled circuits
    res_values["circuit"] = circuit

    return res_values

def _prep_sim(sim_args):
    name = sim_args[0]
    circuit = sim_args[1]
    verbose = sim_args[2]
    res_values = {}
    res_values["name"] = name
    # For simulation, delete all of the measurements
    # original circuit
    nlist = circuit["dag_original"].get_named_nodes("measure")
    for n in nlist:
        circuit["dag_original"]._remove_op_node(n)
    # Skip verification of the reference State to speed things up!
    # if verbose and circuit["dag_reference"] is not None:
    #     # reference circuit
    #     nlist = circuit["dag_reference"].get_named_nodes("measure")
    #     perm = {}
    #     for n in nlist:
    #         nd = circuit["dag_reference"].multi_graph.node[n]
    #         perm[nd["qargs"][0][1]] = nd["cargs"][0][1]
    #         circuit["dag_reference"]._remove_op_node(n)
    #     circuit["perm_reference"] = [perm[ii] for ii in range(len(perm))]
    if circuit["dag_optimized"] is not None:
        # optimized circuit
        nlist = circuit["dag_optimized"].get_named_nodes("measure")
        perm = {}
        for n in nlist:
            nd = circuit["dag_optimized"].multi_graph.node[n]
            perm[nd["qargs"][0][1]] = nd["cargs"][0][1]
            circuit["dag_optimized"]._remove_op_node(n)
        circuit["perm_optimized"] = [perm[ii] for ii in range(len(perm))]

        print("perm opt", circuit["perm_optimized"])
    res_values["circuit"] = circuit
    return res_values

def _qiskit_compiler(dag_circuit, coupling_map=None, gate_costs=None):
    import copy
    from qiskit.mapper import swap_mapper, direction_mapper, cx_cancellation, optimize_1q_gates, Coupling
    from qiskit import qasm, unroll        
    initial_layout = None
    coupling = Coupling(coupling_map)
    compiled_dag, final_layout = swap_mapper(copy.deepcopy(dag_circuit), coupling, initial_layout, trials=40, seed=19)
    # Expand swaps
    basis_gates = "u1,u2,u3,cx,id"  # QE target basis
    program_node_circuit = qasm.Qasm(data=compiled_dag.qasm()).parse()
    unroller_circuit = unroll.Unroller(program_node_circuit,
                                       unroll.DAGBackend(
                                           basis_gates.split(",")))
    compiled_dag = unroller_circuit.execute()
    # Change cx directions
    compiled_dag = direction_mapper(compiled_dag, coupling)
    # Simplify cx gates
    cx_cancellation(compiled_dag)
    # Simplify single qubit gates
    compiled_dag = optimize_1q_gates(compiled_dag)
    # Return the compiled dag circuit
    return compiled_dag


def _get_perm(perm):
    # get the permutation indices for the compiled quantum state given the permutation
    n = len(perm)
    res = np.arange(2**n)
    for num_ind in range(2**n):
        ss_orig = [digit for digit in bin(num_ind)[2:].zfill(n)]
        ss_final = ss_orig.copy()
        for ii in range(n):
            ss_final[n-1-ii] = ss_orig[n-1-perm[ii]]
        res[num_ind] = int(''.join(ss_final),2)
    return list(res)

def _compare_outputs(data_original, data_compiled, permutation, threshold=ERROR_LIMIT):
    # compare the output states of the original and the compiled circuit
    # the bits of the compiled output state are permuted according to permutation before comparision
    if 'quantum_state' in data_original \
            and 'quantum_state' in data_compiled:
        state = data_compiled.get('quantum_state')
        target = data_original.get('quantum_state')
    elif 'quantum_states' in data_original \
            and 'quantum_states' in data_compiled:
        state = data_compiled.get('quantum_states')[0]
        target = data_original.get('quantum_states')[0]
    else:
        return False
    state = state[_get_perm(permutation)]
    fidelity = state_fidelity(target, state)
    #print("fidelity = ", fidelity)
    return abs(fidelity - 1.0) < threshold


def _compose_qobj(qobj_name, test_circuits,
                   backend="local_qasm_simulator",
                   config=None,
                   basis_gates='u1,u2,u3,id,cx',
                   shots=1,
                   max_credits=3,
                   seed=None):
    qobj_out = {"id": qobj_name+"_test_circuits",
                     "config": config,
                     "circuits": []}

    qobj_out["config"] = {"max_credits": max_credits,
                               "backend": backend,
                               "seed": seed,
                               "shots": shots}

    for name, circuit in test_circuits.items():
        # only add the job if there was no error in generating the circuit
        if circuit["dag_"+qobj_name] is not None:
            job = {}
            job["name"] = name
            # config parameters used by the runner
            if config is None:
                config = {}  # default to empty config dict
            job["config"] = copy.deepcopy(config)
            job["config"]["basis_gates"] = basis_gates
            job["config"]["seed"] = seed

            # Add circuit
            dag_circuit = copy.deepcopy(circuit["dag_"+qobj_name])
            job["compiled_circuit"] = dag2json(dag_circuit,
                                               basis_gates=basis_gates)
            job["compiled_circuit_qasm"] = dag_circuit.qasm(qeflag=True)
            qobj_out["circuits"].append(copy.deepcopy(job))

    return qobj_out
