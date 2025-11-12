import os, time, json, glob, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoProcessor, CLIPSegForImageSegmentation

# ----------------- CONFIG -----------------
@dataclass
class Cfg:
    model_name: str = "CIDAS/clipseg-rd64-refined"
    prompt: str = "crack"                 # training text prompt
    target_category_name: str = None      # if set, only masks with this COCO category name are used; else all anns
    resize_in: int = 352                  # CLIPSeg native
    bs: int = 4
    lr: float = 2e-4
    epochs: int = 20
    num_workers: int = 4
    log_every: int = 10
    ckpt_every: int = 500
    amp: bool = True
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    freeze_vision_text: bool = True
    seed: int = 1337
cfg = Cfg()

# =============== COCO HELPERS ===============
def coco_load(json_path: Path):
    data = json.load(open(json_path, "r"))
    imgs = {im["id"]: im for im in data["images"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:  # ignore crowd
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    return data, imgs, anns_by_img, categories

def coco_mask_for_image(img_info: Dict[str, Any],
                        anns: List[Dict[str, Any]],
                        categories: Dict[int, str],
                        target_category_name: str = None) -> np.ndarray:
    """Return a binary mask [H,W] from polygons and/or RLE. If target_category_name is set, keep only those."""
    H, W = img_info["height"], img_info["width"]
    m = np.zeros((H, W), dtype=np.uint8)
    for a in anns:
        if target_category_name is not None:
            cname = categories.get(a.get("category_id"), None)
            if cname != target_category_name:
                continue
        seg = a.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):  # polygons
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
        else:
            # RLE dict
            import pycocotools.mask as maskUtils
            m = np.maximum(m, maskUtils.decode(seg).astype(np.uint8))
    return m

# =============== DATASETS ===============
class COCOSegDataset(Dataset):
    def __init__(self, img_dir: Path, ann_json: Path, target_category_name: str = None, resize_in: int = 352):
        self.img_dir = Path(img_dir)
        self.data, self.imgs_by_id, self.anns_by_img, self.categories = coco_load(ann_json)
        # Build list of (filepath, img_info, anns)
        self.items: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
        for img_id, info in self.imgs_by_id.items():
            fp = self.img_dir / info["file_name"]
            if fp.exists():
                anns = self.anns_by_img.get(img_id, [])
                self.items.append((str(fp), info, anns))
        self.target_category_name = target_category_name
        self.resize_in = resize_in

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, info, anns = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        gt = coco_mask_for_image(info, anns, self.categories, self.target_category_name)

        # Resize to CLIPSeg resolution for training
        img_resized = img.resize((self.resize_in, self.resize_in), Image.BILINEAR)
        gt_resized = cv2.resize(gt, (self.resize_in, self.resize_in), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        return img_resized, gt_resized, img_path

def collate(batch):
    imgs, masks, paths = zip(*batch)
    return list(imgs), np.stack(masks), list(paths)

# =============== METRICS ===============
def iou_dice(pred01, gt01):
    p = pred01.astype(bool); g = gt01.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou  = inter / (union + 1e-8)
    dice = (2*inter) / (p.sum() + g.sum() + 1e-8)
    return float(iou), float(dice)

# =============== TRAIN LOOP ===============
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train(args):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))

    # Data (COCO)
    ds_tr = COCOSegDataset(args.train_images, args.train_ann, cfg.target_category_name, cfg.resize_in)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

    ds_va = None; dl_va = None
    if args.val_images and args.val_ann:
        ds_va = COCOSegDataset(args.val_images, args.val_ann, cfg.target_category_name, cfg.resize_in)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

    # Model + Processor
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(cfg.model_name).to(device)

    # Freeze CLIP encoders, train decoder
    if cfg.freeze_vision_text:
        for n, p in model.named_parameters():
            if n.startswith("vision_model") or n.startswith("text_model") or n.startswith("clip_model") or "transformer" in n:
                p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    bce = nn.BCEWithLogitsLoss()

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_miou = -1.0
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for it, (imgs, masks_np, _) in enumerate(dl_tr):
            t0 = time.perf_counter()

            texts = [cfg.prompt] * len(imgs)
            inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device)
            masks = torch.from_numpy(masks_np).unsqueeze(1).to(device)  # [B,1,352,352]

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(**inputs)
                logits = out.logits.unsqueeze(1)         # [B,1,352,352]
                loss = bce(logits, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

            iter_time = time.perf_counter() - t0
            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
                writer.add_scalar("train/iter_time_sec", iter_time, global_step)
                print(f"[e{epoch} it{it} step{global_step}] loss={loss.item():.4f}  iter_time={iter_time:.3f}s")

            if global_step % cfg.ckpt_every == 0:
                ckpt_dir = out_dir / f"ckpt_step{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir); processor.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint: {ckpt_dir}")

        # ---- Validation (optional) ----
        if dl_va is not None:
            model.eval()
            miou = mdice = 0.0; n = 0
            with torch.no_grad():
                for imgs, masks_np, _ in dl_va:
                    texts = [cfg.prompt] * len(imgs)
                    inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device)
                    probs = torch.sigmoid(model(**inputs).logits).detach().cpu().numpy()
                    preds = (probs >= 0.5).astype(np.uint8)
                    gts = masks_np.astype(np.uint8)
                    for i in range(preds.shape[0]):
                        iou, dice = iou_dice(preds[i], gts[i])
                        miou += iou; mdice += dice; n += 1
            miou /= max(n,1); mdice /= max(n,1)
            writer.add_scalar("val/mIoU", miou, epoch)
            writer.add_scalar("val/mDice", mdice, epoch)
            print(f"[val] mIoU={miou:.4f} mDice={mdice:.4f}")

            if miou > best_miou:
                best_miou = miou
                best_dir = out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir); processor.save_pretrained(best_dir)
                print(f"New best mIoU={best_miou:.4f} → saved {best_dir}")

    print("Training complete.")

# =============== INFERENCE / EVAL ===============
def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)
    processor = AutoProcessor.from_pretrained(ckpt)
    model = CLIPSegForImageSegmentation.from_pretrained(ckpt).to(device).eval()

    img_paths = sorted(glob.glob(str(Path(args.images) / "*.jpg"))) + \
                sorted(glob.glob(str(Path(args.images) / "*.png")))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for ip in tqdm(img_paths, desc="Infer"):
        img = Image.open(ip).convert("RGB")
        inputs = processor(text=[args.prompt], images=[img], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(model(**inputs).logits)[0].detach().cpu().numpy()  # [352,352]
        mask = (cv2.resize(probs, (img.width, img.height)) >= args.thresh).astype(np.uint8)*255
        cv2.imwrite(str(out_dir / (Path(ip).stem + "_pred.png")), mask)

def eval_coco(args):
    """Compute mIoU/mDice against a COCO json using a trained checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)
    processor = AutoProcessor.from_pretrained(ckpt)
    model = CLIPSegForImageSegmentation.from_pretrained(ckpt).to(device).eval()

    data, imgs_by_id, anns_by_img, categories = coco_load(Path(args.ann))
    img_dir = Path(args.images)

    miou = mdice = 0.0; n = 0
    for img_id, info in tqdm(imgs_by_id.items(), desc="Eval COCO"):
        ip = img_dir / info["file_name"]
        if not ip.exists(): 
            continue
        img = Image.open(ip).convert("RGB")
        inputs = processor(text=[args.prompt], images=[img], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(model(**inputs).logits)[0].detach().cpu().numpy()
        pred01 = (cv2.resize(probs, (img.width, img.height)) >= args.thresh).astype(np.uint8)

        gt = coco_mask_for_image(info, anns_by_img.get(img_id, []), categories, cfg.target_category_name)
        i, d = iou_dice(pred01, gt)
        miou += i; mdice += d; n += 1
    miou /= max(n,1); mdice /= max(n,1)
    print(f"COCO Eval — images:{n}  mIoU:{miou:.4f}  mDice:{mdice:.4f}")

# =============== CLI ===============
def parse_args():
    p = argparse.ArgumentParser("CLIPSeg COCO Fine-tune / Infer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--train-images", type=Path, required=True, help="Path to COCO train images folder")
    pt.add_argument("--train-ann", type=Path, required=True, help="Path to COCO instances_train.json")
    pt.add_argument("--val-images", type=Path, default=None, help="Path to COCO val images folder")
    pt.add_argument("--val-ann", type=Path, default=None, help="Path to COCO instances_val.json")
    pt.add_argument("--out-dir", type=Path, default=Path("runs_clipseg_coco"))
    # infer
    pi = sub.add_parser("infer")
    pi.add_argument("--checkpoint", type=Path, required=True, help="Path to saved HF checkpoint dir")
    pi.add_argument("--images", type=Path, required=True, help="Folder with images")
    pi.add_argument("--prompt", type=str, default=cfg.prompt)
    pi.add_argument("--thresh", type=float, default=0.5)
    pi.add_argument("--out-dir", type=Path, default=Path("predictions"))
    # eval
    pe = sub.add_parser("eval")
    pe.add_argument("--checkpoint", type=Path, required=True)
    pe.add_argument("--images", type=Path, required=True)
    pe.add_argument("--ann", type=Path, required=True)
    pe.add_argument("--prompt", type=str, default=cfg.prompt)
    pe.add_argument("--thresh", type=float, default=0.5)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    elif args.cmd == "eval":
        eval_coco(args)


"""
train: python clipseg_ft.py train --train-images cracks.v1i.coco/train --train-ann cracks.v1i.coco/train/_annotations.coco.json --val-images cracks.v1i.coco/valid/ --val-ann cracks.v1i.coco/valid/_annotations.coco.json --out-dir run1
infer: python clipseg_ft.py infer --checkpoint run1/ckpt_step500 --images cracks.v1i.coco/test
eval:  python clipseg_ft.py eval --checkpoint run1/ckpt_step500 --images cracks.v1i.coco/test --ann cracks.v1i.coco/test/_annotations.coco.json
"""
