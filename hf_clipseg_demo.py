import os, json, glob
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import cv2

from transformers import AutoProcessor, CLIPSegForImageSegmentation

# ---------------- CONFIG ----------------
# Point these to your Roboflow export:
# If you exported PNG masks:
PNG_ROOT   = Path("cracks.v1i.coco")             # dataset root
PNG_SPLIT  = "test"                             # or "train"/"valid"
PNG_IMG_DIR  = PNG_ROOT / PNG_SPLIT / "images"
PNG_MASK_DIR = PNG_ROOT / PNG_SPLIT / "labels"  # binary masks (0/255)

# If you exported COCO segmentation:
COCO_ROOT     = Path("cracks.v1i.coco")
COCO_IMG_DIR  = COCO_ROOT / PNG_SPLIT
COCO_ANN_JSON = COCO_IMG_DIR / "_annotations.coco.json"

USE_PNG_MASKS = False        # False -> use COCO JSON
PROMPT        = "crack"     # try: "concrete crack", "pavement crack"
THRESH        = 0.5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = Path("hf_clipseg_outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR       = Path("gt"); GT_DIR.mkdir(parents=True, exist_ok=True)

# -------------- HELPERS -----------------
def load_gt_mask_png(img_path: Path):
    mask_path = PNG_MASK_DIR / (img_path.stem + ".png")
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    return (m > 127).astype(np.uint8)

def coco_build_index(json_path: Path):
    data = json.load(open(json_path, "r"))
    imgs = {im["id"]: im for im in data["images"]}
    by_img = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1: 
            continue
        by_img.setdefault(ann["image_id"], []).append(ann)
    return data, imgs, by_img

def coco_mask_for_image(img_info, anns):
    h, w = img_info["height"], img_info["width"]
    m = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        seg = a.get("segmentation", None)
        if seg is None:
            continue
        if isinstance(seg, list):  # polygons
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
        else:
            import pycocotools.mask as maskUtils
            m = np.maximum(m, maskUtils.decode(seg).astype(np.uint8))
    return m

def iou_dice(pred01, gt01):
    p = pred01.astype(bool); g = gt01.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou  = inter / (union + 1e-8)
    dice = (2*inter) / (p.sum() + g.sum() + 1e-8)
    return float(iou), float(dice)

# -------------- MAIN --------------------
def main():
    print(f"Device: {DEVICE}")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE).eval()

    if USE_PNG_MASKS:
        img_paths = sorted(glob.glob(str(PNG_IMG_DIR / "*.jpg"))) + \
                    sorted(glob.glob(str(PNG_IMG_DIR / "*.png")))
        coco_data = coco_imgs = anns_by_img = None
    else:
        img_paths = sorted(glob.glob(str(COCO_IMG_DIR / "*.jpg"))) + \
                    sorted(glob.glob(str(COCO_IMG_DIR / "*.png")))
        coco_data, coco_imgs, anns_by_img = coco_build_index(COCO_ANN_JSON)

    metrics = []
    texts = [PROMPT]  # single prompt; you can pass multiple (["concrete crack", "asphalt crack", ...])

    for ip in tqdm(img_paths, desc="CLIPSeg (HF)"):
        img_pil = Image.open(ip).convert("RGB")

        # HF input: duplicate image for each text prompt
        inputs = processor(text=texts, images=[img_pil]*len(texts), padding=True, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits  = outputs.logits  # [num_texts, 352, 352]
            probs   = torch.sigmoid(logits).detach().cpu().numpy()[0]  # for our single prompt

        # resize back to original image size
        probs_up = cv2.resize(probs, (img_pil.width, img_pil.height), interpolation=cv2.INTER_CUBIC)
        pred01   = (probs_up >= THRESH).astype(np.uint8)

        # load GT
        if USE_PNG_MASKS:
            gt01 = load_gt_mask_png(Path(ip))
        else:
            name = Path(ip).name
            img_info = next((im for im in coco_data["images"] if im["file_name"] == name), None)
            if img_info is None:
                print(f"[WARN] COCO missing for {name}; skip")
                continue
            anns = anns_by_img.get(img_info["id"], [])
            gt01 = coco_mask_for_image(img_info, anns)

        i, d = iou_dice(pred01, gt01)
        metrics.append((ip, i, d))


        # save mask
        out_mask = (gt01*255).astype(np.uint8)
        cv2.imwrite(str(GT_DIR / (Path(ip).stem + "_clipseg.png")), out_mask)

        # save mask
        out_mask = (pred01*255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / (Path(ip).stem + "_clipseg.png")), out_mask)

    if metrics:
        miou = sum(x[1] for x in metrics)/len(metrics)
        mdice = sum(x[2] for x in metrics)/len(metrics)
        print(f"Images: {len(metrics)}  mIoU: {miou:.4f}  mDice: {mdice:.4f}")

        with open(OUT_DIR / "metrics.csv", "w") as f:
            f.write("image_path,IoU,Dice\n")
            for p,i,d in metrics:
                f.write(f"{p},{i:.6f},{d:.6f}\n")
        print(f"Saved masks + metrics to {OUT_DIR}")

if __name__ == "__main__":
    main()
