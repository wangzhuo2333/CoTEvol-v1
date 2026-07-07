"""Grid Search for Tuning"""

import os
import sys
import itertools as it
from loguru import logger
from multiprocessing import Pool

def run_process(proc):
    os.system(proc)


hyperparameter_grid = {
    # "data_type": ["long", "short", "evol"],
    "seed": [42, 1314, 2025],
}

device = sys.argv[1]
data_name = sys.argv[2]
data_type = sys.argv[3]
data_suffix = sys.argv[4]


cmds = []
hyper_parameter = hyperparameter_grid
for parameter in it.product(*list(hyper_parameter.values())):
    specific_parameter_dict = {key: parameter[list(hyper_parameter.keys()).index(key)]
                               for key in list(hyper_parameter.keys())}

    cmd = f'CUDA_VISIBLE_DEVICES={device} python run_rm.py '
    options = [
        "--data_type", f"{data_type}",
        "--output_dir", "/code/Research_with_user/reasoning/GA/reward_score/v1.1/",
        "--data_dir", f"/extrahome0/user/output/s1k_v/{data_name}_{data_type}_{data_suffix}.pt"
    ]
    for key, value in specific_parameter_dict.items():
        options.extend(["--" + key, str(value)])

    one_cmd = cmd + " ".join(options)
    one_cmd += " & wait"
    cmds.append(one_cmd)

run_process("sleep 2s")
logger.warning(f"run {len(cmds)} grid-search tasks for {data_name}")
# print(cmds[0])
# run_process(cmds[0])  # debug
pool = Pool(processes=1)
pool.map(run_process, cmds)

