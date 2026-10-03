import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config, data, trainer
from models import UNetGenerator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{config.OUT_DIR}/runs/final/best.pt")
ap.add_argument("--split", default="test")
args = ap.parse_args()

dev = trainer.DEVICE
ck = torch.load(args.ckpt, map_location=dev)
cfg = ck["cfg"]
G = UNetGenerator(cfg["base_ch"], cfg["emb_dim"], cfg["dropout"]).to(dev)
G.load_state_dict(ck["G"]); G.eval()

ds = data.get_dataset(args.split)
loader = DataLoader(ds, batch_size=32, shuffle=False)
L1, PS, SS, FK = [], [], [], []
with torch.no_grad():
    for p, s, st in loader:
        p, s, st = p.to(dev), s.to(dev), st.to(dev)
        f = G(p, st)
        l1, ps, ss = trainer.per_sample_metrics(f, s)
        L1.append(l1.cpu()); PS.append(ps.cpu()); SS.append(ss.cpu()); FK.append(f.cpu())
L1, PS, SS, FK = map(torch.cat, (L1, PS, SS, FK))
styles = np.array(ds.styles)


def summ(mask):
    return {k: {"mean": float(v[mask].mean()), "std": float(v[mask].std())}
            for k, v in (("l1", L1), ("psnr", PS), ("ssim", SS))}


res = {"split": args.split, "n": len(ds), "epoch": ck["epoch"],
       "overall": summ(np.ones(len(ds), bool))}
for k in range(config.NUM_STYLES):
    res[f"style_{k}"] = {"n": int((styles == k).sum()), **summ(styles == k)}
json.dump(res, open(f"{config.RESULTS_DIR}/task4_{args.split}_metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))


def grid_for(idx, name):
    idx = list(idx)
    p = torch.stack([ds[i][0] for i in idx]); s = torch.stack([ds[i][1] for i in idx])
    save_image(trainer.triplet_grid(p, s, FK[idx]), f"{config.RESULTS_DIR}/{name}")


grid_for(range(12), f"task4_{args.split}_samples.png")
order = torch.argsort(SS)
grid_for(order[:6].tolist(), f"task4_{args.split}_worst6.png")      # failure cases
grid_for(order[-6:].tolist(), f"task4_{args.split}_best6.png")
print("worst-6 indices:", order[:6].tolist(), "styles:", styles[order[:6].numpy()].tolist())
