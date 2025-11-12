#!/usr/bin/env python3
"""
Make an HTML side-by-side gallery from three folders:
- train/         (original images)
- gt/            (ground-truth masks)
- predictions/   (predicted masks)

Pairs items by filename *stem* (e.g., foo.jpg ↔ foo.png).
Optionally, provide suffixes for GT and Pred if files are named like foo_mask.png, foo_pred.png.

Usage:
    python make_gallery.py \
        --train-dir path/to/train/images \
        --gt-dir path/to/gt \
        --pred-dir path/to/predictions \
        --out gallery.html \
        --gt-suffix "" \
        --pred-suffix "" \
        --thumb-max 420 \
        --title "Crack Segmentation: Train vs GT vs Pred"

Notes:
- The script embeds images as base64 thumbnails (single self-contained HTML).
- Supported extensions: .png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp
"""

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def scan_images(folder: Path) -> Dict[str, Path]:
    """
    Return dict mapping lowercase stem -> full path for image files in a folder.
    If multiple extensions exist for same stem, the first encountered wins.
    """
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stem = p.stem.lower()
            if stem not in out:
                out[stem] = p
    return out

def find_with_suffix(folder: Path, stem: str, suffix: str) -> Optional[Path]:
    """
    Look for a file in folder whose stem matches (stem + suffix),
    trying all supported extensions.
    """
    target_stem = (stem + suffix).lower()
    for ext in IMG_EXTS:
        candidate = folder / f"{target_stem}{ext}"
        if candidate.exists():
            return candidate
    # If not found, also try plain stem (no suffix), handy when suffix not actually present
    for ext in IMG_EXTS:
        candidate = folder / f"{stem.lower()}{ext}"
        if candidate.exists():
            return candidate
    return None

def make_thumbnail_b64(path: Optional[Path], max_side: int) -> Tuple[str, str]:
    """
    Open an image, resize to fit within max_side (keeping aspect), and return (data_uri, info_text).
    If path is None or not found, return a placeholder.
    """
    if path is None:
        # transparent 1x1 png
        return ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=", "missing")
    try:
        im = Image.open(path).convert("RGBA")
        w, h = im.size
        scale = 1.0
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), Image.BILINEAR)
        with io.BytesIO() as buf:
            # Use PNG to avoid JPEG mask artifacts
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return (f"data:image/png;base64,{b64}", f"{path.name} ({w}×{h}→{new_w}×{new_h})")
    except Exception as e:
        # return a red 8x8 error block
        err = Image.new("RGBA", (8, 8), (200, 40, 40, 255))
        with io.BytesIO() as buf:
            err.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return (f"data:image/png;base64,{b64}", f"error: {path} ({e})")

def build_rows(
    train_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    gt_suffix: str,
    pred_suffix: str,
    thumb_max: int,
) -> List[Dict]:
    """
    Iterate over train images and find matching GT and Pred.
    Returns list of dicts: { 'key':stem, 'train':..., 'gt':..., 'pred':..., 'meta':... }
    """
    train_map = scan_images(train_dir)
    rows: List[Dict] = []

    for stem, train_path in sorted(train_map.items()):
        gt_path = find_with_suffix(gt_dir, stem, gt_suffix) if gt_dir else None
        pred_path = find_with_suffix(pred_dir, stem, pred_suffix) if pred_dir else None

        data_train, info_train = make_thumbnail_b64(train_path, thumb_max)
        data_gt, info_gt = make_thumbnail_b64(gt_path, thumb_max)
        data_pred, info_pred = make_thumbnail_b64(pred_path, thumb_max)

        rows.append({
            "key": stem,
            "train_src": data_train,
            "gt_src": data_gt,
            "pred_src": data_pred,
            "train_info": info_train,
            "gt_info": info_gt,
            "pred_info": info_pred,
            "train_path": str(train_path) if train_path else "",
            "gt_path": str(gt_path) if gt_path else "",
            "pred_path": str(pred_path) if pred_path else "",
        })
    return rows

def write_html(out_path: Path, title: str, rows: List[Dict]):
    css = f"""
    <style>
      body {{
        margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji";
        background: #0b0c10; color: #e5e7eb;
      }}
      header {{
        position: sticky; top: 0; z-index: 10;
        background: #111827cc; backdrop-filter: blur(6px);
        padding: 12px 16px; border-bottom: 1px solid #1f2937;
        display: grid; grid-template-columns: 1fr 300px; gap: 12px; align-items: center;
      }}
      h1 {{ font-size: 18px; margin: 0; }}
      .search input {{
        width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #374151; background: #0f172a; color: #e5e7eb;
      }}
      .legend {{
        display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; padding: 10px 16px; font-size: 13px; color:#93c5fd;
      }}
      .grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(780px, 1fr)); gap: 16px; padding: 16px;
      }}
      .card {{
        border: 1px solid #1f2937; border-radius: 12px; background: #0f172a; overflow: hidden;
      }}
      .card-header {{
        padding: 10px 12px; border-bottom: 1px solid #1f2937; font-size: 13px; color:#a5b4fc; display:flex; justify-content:space-between; gap:10px; flex-wrap: wrap;
      }}
      .triptych {{
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; background:#111827;
      }}
      figure {{
        margin: 0; padding: 10px; border-right: 1px solid #1f2937;
      }}
      figure:last-child {{ border-right: none; }}
      figcaption {{
        font-size: 12px; color: #9ca3af; margin-top: 6px; word-break: break-word;
      }}
      img {{
        width: 100%; height: auto; display: block; background: #0b0c10;
        border-radius: 8px; border: 1px solid #1f2937;
      }}
      .meta {{
        display:flex; gap:12px; flex-wrap: wrap; font-size: 11px; color:#9ca3af;
      }}
      .tag {{ padding:4px 8px; background:#0b1220; border:1px solid #1f2937; border-radius:999px; }}
      .hidden {{ display: none !important; }}
      footer {{ padding: 20px; color:#6b7280; font-size:12px; text-align:center; }}
      a {{ color:#60a5fa; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
    """

    js = """
    <script>
    function setupSearch() {
      const input = document.getElementById('search');
      const items = document.querySelectorAll('.card');
      input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        items.forEach(card => {
          const key = card.dataset.key;
          if (!q || key.includes(q)) card.classList.remove('hidden');
          else card.classList.add('hidden');
        });
      });
    }
    window.addEventListener('DOMContentLoaded', setupSearch);
    </script>
    """

    # Build cards
    cards = []
    for r in rows:
        card = f"""
        <div class="card" data-key="{r['key']}">
          <div class="card-header">
            <div><strong>{r['key']}</strong></div>
            <div class="meta">
              <span class="tag">train: {os.path.basename(r['train_path']) or '—'}</span>
              <span class="tag">gt: {os.path.basename(r['gt_path']) or '—'}</span>
              <span class="tag">pred: {os.path.basename(r['pred_path']) or '—'}</span>
            </div>
          </div>
          <div class="triptych">
            <figure>
              <img src="{r['train_src']}" alt="train {r['key']}"/>
              <figcaption>train • {r['train_info']}</figcaption>
            </figure>
            <figure>
              <img src="{r['gt_src']}" alt="gt {r['key']}"/>
              <figcaption>ground truth • {r['gt_info']}</figcaption>
            </figure>
            <figure>
              <img src="{r['pred_src']}" alt="pred {r['key']}"/>
              <figcaption>prediction • {r['pred_info']}</figcaption>
            </figure>
          </div>
        </div>
        """
        cards.append(card)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  {css}
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="search"><input id="search" placeholder="Search by filename… (live filter)"/></div>
  </header>
  <div class="legend">
    <div>Left: Train (original)</div>
    <div>Center: Ground Truth</div>
    <div>Right: Prediction</div>
  </div>
  <main class="grid">
    {''.join(cards)}
  </main>
  <footer>Generated by make_gallery.py • {len(rows)} items</footer>
  {js}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} rows)")

def main():
    ap = argparse.ArgumentParser("Build an HTML gallery for Train vs GT vs Predictions")
    ap.add_argument("--train-dir", type=Path, required=True, help="Folder with original images")
    ap.add_argument("--gt-dir", type=Path, required=True, help="Folder with ground-truth masks")
    ap.add_argument("--pred-dir", type=Path, required=True, help="Folder with predicted masks")
    ap.add_argument("--gt-suffix", type=str, default="", help="Suffix appended to stem for GT files, e.g. _mask")
    ap.add_argument("--pred-suffix", type=str, default="", help="Suffix appended to stem for Pred files, e.g. _pred")
    ap.add_argument("--thumb-max", type=int, default=420, help="Max thumbnail side in pixels")
    ap.add_argument("--title", type=str, default="Segmentation Gallery (Train • GT • Prediction)")
    ap.add_argument("--out", type=Path, default=Path("gallery.html"))
    args = ap.parse_args()

    rows = build_rows(args.train_dir, args.gt_dir, args.pred_dir, args.gt_suffix, args.pred_suffix, args.thumb_max)
    write_html(args.out, args.title, rows)

if __name__ == "__main__":
    main()
