# GPU Preempter

## Description
If you are using a shared GPU server with your lab-mates and you cannot preempt GPUs because of their annoying jobs, you can try this!
It automatically monitors the GPU availability via `nvidia-smi` and deploys your job once there are enough available GPUs on this server.

## Usage
1. Write a json config file for your job like this
```json
{
    "JobCommand": "/usr/bin/python3 ./sample_job.py 4",
    "GPURequired": 4,
    "OnlyUseFreeGPU": false,
    "GPUMemRequiredTotal": 4096,
    "MinPerGPUMemRequired": 1024
}
```
Here,
- `JobCommand` is the command to run your program
- `GPURequired` is the number of GPUs you want.
- If `OnlyUseFreeGPU` is set to `true`, then it will only run your job if there are `GPURequired` totally free GPUs (i.e., no one use them). Otherwise, your job may be co-located with your labmates' jobs, and you need to specify the following two parameters:
    * `GPUMemRequiredTotal`: total GPU memory in MiB you require
    * `MinPerGPUMemRequired`: each GPU should have at least this amount of available memory

2. Run your job via
```shell
python preempter.py --config <PATH_TO_YOUR_CONFIG_FILE> --monitor-interval <INTERVAL_OF_MONITOR>
```
Then it will launch a agent to wait for resources and automatically run your job once GPUs are available.
