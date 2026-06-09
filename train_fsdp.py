import torch
import json
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from model import get_model
from dataset import ToyDataset
from metrics import Timer, gpu_util, mem


def run():
    model = get_model().cuda()
    model = FSDP(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(ToyDataset(), batch_size=2)

    for step, batch in enumerate(loader):
        batch = {k: v.cuda() for k, v in batch.items()}

        with Timer() as t:
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()

        log = {
            "step": step,
            "time_ms": t.dt,
            "gpu": gpu_util(),
            "mem": mem(),
            "loss": loss.item()
        }

        print(json.dumps(log))

        if step >= 50:
            break


if __name__ == "__main__":
    run()
