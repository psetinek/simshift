import argparse
import itertools
import os
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime

import yaml


def parse_value(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def main():
    parser = argparse.ArgumentParser(
        description="Job launcher for multiple experiments."
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="Maximum number of concurrent runs",
    )
    parser.add_argument(
        "--gpus", type=str, help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs", help="Path to config folder."
    )
    parser.add_argument(
        "parameters", nargs="+", help="Parameters in key=value1,value2,... format"
    )
    args = parser.parse_args()

    # GPU management setup
    gpu_pool = deque()
    if args.gpus:
        gpu_pool = deque(map(str, args.gpus.split(",")))
        args.max_concurrent = min(args.max_concurrent, len(gpu_pool))

    # Parse parameters
    params = {}
    for param in args.parameters:
        if "=" not in param:
            print(f"Invalid parameter format: {param}. Skipping.")
            continue
        key, values_str = param.split("=", 1)
        values = [parse_value(v) for v in values_str.split(",")]
        params[key] = values

    # Generate all combinations
    keys = params.keys()
    values = params.values()
    combinations = [
        dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)
    ]

    # Create multirun directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    multirun_dir = os.path.join("multirun", timestamp)
    os.makedirs(multirun_dir, exist_ok=True)

    # Save base configs
    shutil.copytree(args.config_path, os.path.join(multirun_dir, "configs"))

    # Prepare and execute runs
    processes = []
    index = 0

    try:
        while index < len(combinations) or len(processes) > 0:
            # Start new processes
            while index < len(combinations) and len(processes) < args.max_concurrent:
                if args.gpus and not gpu_pool:
                    break  # Wait for GPUs to become available

                combo = combinations[index]
                run_dir = os.path.join(multirun_dir, f"run_{index:04d}")
                os.makedirs(run_dir, exist_ok=True)

                # Save parameters
                override_path = os.path.join(run_dir, "override.yaml")
                with open(override_path, "w") as f:
                    yaml.dump(combo, f)

                # Build command
                cmd = (
                    ["python", "main.py"]
                    + [f"-cp={os.path.join(multirun_dir, 'configs')}"]
                    + [f"{k}={v}" for k, v in combo.items()]
                )

                # Assign GPU if available
                env = os.environ.copy()
                gpu_id = None
                if args.gpus:
                    gpu_id = gpu_pool.popleft()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    print(f"Assigned GPU {gpu_id} to run {index}")

                # Start process
                log_path = os.path.join(run_dir, "run.out")
                with open(log_path, "w") as log_file:
                    proc = subprocess.Popen(
                        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env
                    )
                    processes.append((proc, gpu_id))
                    print(f"Started run {index} with PID {proc.pid}")

                index += 1
                time.sleep(10)

            # Check completed processes
            new_processes = []
            for proc, gpu_id in processes:
                if proc.poll() is None:  # Still running
                    new_processes.append((proc, gpu_id))
                else:  # Process finished
                    if gpu_id is not None:
                        gpu_pool.append(gpu_id)
                        print(f"Released GPU {gpu_id}")
                    print(f"Process {proc.pid} finished")
            processes = new_processes

            time.sleep(5)

        print("All runs completed.")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating all processes.")
        for proc, gpu_id in processes:
            proc.terminate()
            if gpu_id is not None:
                gpu_pool.append(gpu_id)
        for proc, _ in processes:
            proc.wait()
        print("Cleanup completed.")


if __name__ == "__main__":
    main()
