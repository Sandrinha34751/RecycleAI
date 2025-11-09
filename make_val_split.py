import os, shutil, random
from pathlib import Path

def make_val_split(root='dataset', train_folder='train', val_folder='val', val_frac=0.15, seed=42):
    random.seed(seed)
    root = Path(root)
    train_dir = root / train_folder
    val_dir = root / val_folder
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir.resolve()}")
    
    val_dir.mkdir(parents=True, exist_ok=True)
    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError("Nenhuma subpasta de classes encontrada em dataset/train")
    for cls in classes:
        tgt_cls = val_dir / cls.name
        tgt_cls.mkdir(parents=True, exist_ok=True)
        imgs = [p for p in cls.iterdir() if p.is_file()]
        n_val = max(1, int(len(imgs) * val_frac)) if imgs else 0
        val_imgs = random.sample(imgs, n_val) if n_val > 0 else []
        for p in val_imgs:
            dest = tgt_cls / p.name
            
            shutil.copy2(p, dest)
        print(f"class={cls.name}: total={len(imgs)}, val={len(val_imgs)}")

if __name__ == "__main__":
    make_val_split(root='dataset', train_folder='train', val_folder='val', val_frac=0.15, seed=42)
    print("Split criado (copiado). Verifique dataset/val/")
