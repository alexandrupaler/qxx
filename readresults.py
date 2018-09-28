import json

for seedi in range(251):
    #fname = "./run50_reparat/run_once_results.json_%d" % seedi
    fname = "./run300_no_att/run_once_results.json_%d" % seedi
    with open(fname, "r") as f:
        results = json.load(f)
    for key in results.keys():
        print(seedi, key, results[key]["cost_optimized"], results[key]["cost_reference"],
              results[key]["optimizer_time"], results[key]["reference_time"])