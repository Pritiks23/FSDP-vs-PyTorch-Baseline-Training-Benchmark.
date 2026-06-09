import torch
import time
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_util():
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

def mem():
    return torch.cuda.memory_allocated() / 1e6

class Timer:
    def __enter__(self):
        self.s = time.time()
        return self

    def __exit__(self, *args):
        self.e = time.time()
        self.dt = (self.e - self.s) * 1000
