import argparse
import multiprocessing as mp
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--num_episodes', type=int, help='num. episodes', default=10000)
parser.add_argument('--cores', type=int, help='num. cores', default=8)
parser.add_argument('--env', type=str, help='env. name', default="CSTR")
parser.add_argument('--algo', type=str, help='algo. name', default="CQL")
parser.add_argument('--iter', type=int, help='num. iter.', default=0)


def run(cmd):
    print(cmd)
    os.system(cmd)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    cmds = []
    
    episodes_step = args.num_episodes // args.cores
    for env in args.env.split(","):
        for start_episodes in range(0, args.num_episodes, episodes_step):
            seed = random.randint(0, 10000)
            cmd_prefix = f"python data_collection.py --seed {seed} --algo {args.algo} --start_episodes {start_episodes} --end_episodes {start_episodes+episodes_step} --env {env} --iter {args.iter} "
            if args.amlt: cmd_prefix += "--amlt "
            cmds.append(cmd_prefix)
    jobs = []
    for i, cmd in enumerate(cmds):
        p = mp.Process(target=run, args=(cmd,))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()