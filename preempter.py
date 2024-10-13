import subprocess
import multiprocessing
import json
import datetime
import time
import logging
import torch
import argparse
import uuid
import numpy as np
from typing import Dict
from multiprocessing import shared_memory
from monitor import start_monitor


def check_shared_memory_exists(name):
    try:
        shm = multiprocessing.shared_memory.SharedMemory(name=name)
        shm.close()
        return True
    except FileNotFoundError:
        return False


def run_job(config: Dict, shm_array: np.ndarray, interval: int) -> int:
    gpu_required = config['GPURequired']
    job_cmd = config['JobCommand']
    only_use_free_gpu = config['OnlyUseFreeGPU']
    avail_gpus = config.get("AvailGPUs", np.arange(len(shm_array)))
    unavil_gpus = []
    for i in range(len(shm_array)):
        if i not in avail_gpus:
            unavil_gpus.append(i)
    unavil_gpus = np.array(unavil_gpus)
    # Wait for resources
    while True:
        try:
            time.sleep(interval)
            array_copy = shm_array.copy()
            if len(unavil_gpus) != 0:
                array_copy[unavil_gpus, :] = 1
            free_devices = np.where(array_copy[:, 0] == 0)[0]
            logging.critical(f"Free GPUs: {free_devices}")
            if only_use_free_gpu:
                if len(free_devices) >= gpu_required:
                    break
            else:
                free_mem = [(i, array_copy[i, 2] - array_copy[i, 1]) for i in range(len(array_copy))]
                free_mem.sort(key=lambda x: x[1], reverse=True)
                if sum([i[1] for i in free_mem[:gpu_required]]) >= config['GPUMemRequiredTotal']\
                   and min([i[1] for i in free_mem[:gpu_required]]) >= config['MinPerGPUMemRequired']:
                    free_devices = [i[0] for i in free_mem[:gpu_required]]
                    logging.critical(f"GPU set {free_devices} satisfies the memory requirement, will deploy on them")
                    break
        except KeyboardInterrupt:
            logging.critical("Stop waiting for resources due to interrupt")
            return -1

    dev_list = ','.join([str(i) for i in free_devices[:gpu_required]])
    run_cmd = f"CUDA_VISIBLE_DEVICES={dev_list} " + job_cmd
    logging.critical(f"Job is successfully deployed at {datetime.datetime.now()} on GPUs: {dev_list}")
    proc = subprocess.run(run_cmd, shell=True)
    return proc.returncode


def main():
    # Parse the job config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./job.json')
    parser.add_argument("--monitor-interval", type=int, default=3)
    args = parser.parse_args()
    interval = args.monitor_interval
    shm_name = "monitor_shm_" + uuid.uuid4().hex[:16]
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Start the nvidia-smi monitor if it not exists
    if not check_shared_memory_exists(shm_name):
        monitor_process = multiprocessing.Process(
            target=start_monitor, args=(shm_name, interval))
        monitor_process.start()
        logging.critical("Creating a new monitor...")
    else:
        logging.error("WTF, shared memory exists!")
        return

    time.sleep(5) # wait for the monitor to initialize
    all_devices = torch.cuda.device_count()
    shm = shared_memory.SharedMemory(shm_name, create=False, size=all_devices * 3 * 4)
    shm_array = np.frombuffer(shm.buf, np.int32, all_devices * 3).reshape((all_devices, 3))

    # Run the job
    retval = run_job(config, shm_array, interval)
    logging.critical(f"Job terminates with exit code {retval}")
    del shm_array
    monitor_process.terminate()
    logging.critical("Preempter shutdown gracefully!")


if __name__ == '__main__':
    main()
