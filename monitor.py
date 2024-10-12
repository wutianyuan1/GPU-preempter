import subprocess
import torch
import logging
import re
import time
import numpy as np
from multiprocessing import shared_memory


class GPUStatsMonitor(object):
    def __init__(self, monitor_shm_name: str = "monitor_shm", monitor_interval: int = 5) -> None:
        self.monitor_interval = monitor_interval
        available_devices = torch.cuda.device_count()
        self.shm = shared_memory.SharedMemory(
            monitor_shm_name, create=True, size=available_devices * 3 * 4)
        # shm_array[device_id] = [#Processes on it, Used GPU memory (MiB), Total GPU memory (MiB)]
        self.shm_array = np.frombuffer(
            self.shm.buf, np.int32, available_devices * 3).reshape((available_devices, 3))

    def __del__(self) -> None:
        del self.shm_array
        self.shm.close()
        self.shm.unlink()

    def get_gpu_stats_str(self) -> str:
        proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE)
        proc.wait()
        return proc.stdout.read().decode()

    def start_monitoring(self) -> None:
        while True:
            tmp_array = np.zeros_like(self.shm_array, dtype=np.int32)
            output_list = self.get_gpu_stats_str().split('\n')
            line_id = 0
            dev_id = 0
            while line_id < len(output_list):
                if 'Processes:' not in output_list[line_id]:
                    if 'MiB' in output_list[line_id]:
                        total_mem = int(re.findall(r"/\ *(\d+)MiB", output_list[line_id])[0])
                        tmp_array[dev_id, 2] = total_mem
                        dev_id += 1
                    line_id += 1
                else:
                    break
            if line_id == len(output_list) - 1:
                logging.warning("Processes not found, will continue.")
                continue
            while line_id < len(output_list):
                # For each active process, nvidia-smi outputs its memory usage in MiB
                # Therefore, we can use the keyword MiB to identify if it is a job line.
                if 'MiB' not in output_list[line_id]:
                    line_id += 1
                    continue
                used_device = int(re.findall(r"\|\ *(\d+)", output_list[line_id])[0])
                used_memory = int(re.findall(r"(\d+)MiB", output_list[line_id])[0])
                tmp_array[used_device, 0] += 1
                tmp_array[used_device, 1] += used_memory
                line_id += 1
            # Clear the previous output
            self.shm_array[:] = tmp_array
            time.sleep(self.monitor_interval)


def start_monitor(monitor_shm_name: str = "monitor_shm", monitor_interval: int = 5) -> None:
    monitor = GPUStatsMonitor(monitor_shm_name, monitor_interval)
    monitor.start_monitoring()


if __name__ == '__main__':
    start_monitor()
