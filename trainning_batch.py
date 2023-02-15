import argparse
import multiprocessing as mp
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--algo', type=str, help='algos', default="CQL")


def run(cmd):
    print(cmd)
    os.system(cmd)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    cmds = []
    for algo in args.algo.split(","):
        cmd_prefix = f"python rl_trainning.py --algo {algo} "
        if args.amlt: cmd_prefix += "--amlt "
        cmds.append(cmd_prefix)
    device_count = torch.cuda.device_count()
    jobs = []
    for i, cmd in enumerate(cmds):
        device_idx = i % device_count
        cmd += " --device %i"%device_idx
        p = mp.Process(target=run, args=(cmd,))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
