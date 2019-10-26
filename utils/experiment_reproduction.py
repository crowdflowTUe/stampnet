import os
import json
import pprint
import imp


import argparse
parser = argparse.ArgumentParser(description="Reproduce an experiment.")
parser.add_argument('file', help="The file name of the experiment.")
parser.add_argument('nr', help="Number of the experiment.")
parser.add_argument('-e', 
                    '--exp', 
                    help="Name of the experiment. Defaults as file name.", 
                    nargs="?")

args = parser.parse_args()

# Experiment information
file_name = args.file
if args.exp:
    experiment_name = args.exp
else:
    experiment_name = file_name
experiment_nr = args.nr

print(file_name, experiment_name, experiment_nr)

exp_prefix = os.path.join("runs", experiment_name, experiment_nr)

config_file = os.path.join(exp_prefix, "config.json")
run_file = os.path.join(exp_prefix, "run.json")

with open(config_file) as cf:
    config_params = json.load(cf)
    
with open(run_file) as rf:
    run_params = json.load(rf)

print("Parameters of this run: ")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config_params)

# Retrieving the source file as noted in the config file
for source_file, snapshot in run_params["experiment"]["sources"]:
    if file_name in source_file:
        source_path = os.path.join(exp_prefix, "..", snapshot)
        exp_file = imp.load_source(file_name, source_path)

print("Running experiment...")
# Same configuration as the experiment.
# These parameters could have been changed in the command line.
exp_file.ex.add_config(config_file)

# Running the experiment.
exp_file.ex.run()
