import torch
import sys

n_gpus = int(sys.argv[1])
print(f"Run on {n_gpus} GPUs")

tensors = []
for i in range(n_gpus):
    tensors.append(torch.rand((4096, 4096), dtype=torch.float, device=f'cuda:{i}'))

while True:
    try:
        for i in range(n_gpus):
            x = torch.rand((4096, 4096), dtype=torch.float, device=f'cuda:{i}')
            ret = torch.mm(x, tensors[i])
    except KeyboardInterrupt:
        print("Shutdown gracefully!")
        break
