"""
Verifiziert SentenceTransformer-Modellordner nach einem Dateitransfer.

Hintergrund: Beim Transfer vom 18.5. wurden die safetensors von M3/M4/M5
korrumpiert (NaN in ~0.4% der Werte = Erwartungswert fuer Zufallsbytes) —
die Modelle lieferten NaN-Embeddings und die Eval degenerierte still.

Checks pro Modell:
  1. NaN-Scan aller Tensoren in allen model.safetensors
  2. Probe-Encoding: Embeddings endlich, nicht konstant (Cosine zweier
     verschiedener Saetze deutlich < 1)

Verwendung:
    python verify_models.py                          # M3/M4/M5 aus config.MODELS (HF-IDs)
    python verify_models.py PFAD|HF-ID [...]         # explizite Modellordner oder HF-IDs

HF-IDs werden ueber huggingface_hub.snapshot_download in den lokalen HF-Cache
geholt und dort geprueft.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

PROBE = ["Zuerich ist eine grosse Stadt in der Schweiz.",
         "Der Eiger ist ein Berg im Berner Oberland."]


def check_safetensors(model_dir: Path) -> tuple[bool, str]:
    files = sorted(model_dir.rglob("*.safetensors"))
    if not files:
        return False, "keine safetensors gefunden"
    for path in files:
        nan_tensors = total = 0
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                total += 1
                if torch.isnan(f.get_tensor(k)).any():
                    nan_tensors += 1
        if nan_tensors:
            return False, f"{path.relative_to(model_dir)}: NaN in {nan_tensors}/{total} Tensoren"
    return True, f"{len(files)} safetensors ohne NaN"


def check_encoding(model_dir: Path) -> tuple[bool, str]:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(model_dir))
    e = m.encode(PROBE, convert_to_tensor=True, show_progress_bar=False)
    if not torch.isfinite(e).all():
        return False, "Probe-Encoding enthaelt NaN/Inf"
    cos = F.cosine_similarity(e[0].unsqueeze(0), e[1].unsqueeze(0)).item()
    if cos > 0.999:
        return False, f"Embeddings (nahezu) konstant: cos={cos:.5f}"
    return True, f"Probe-Encoding ok (dim={e.shape[1]}, cos={cos:.3f})"


def _resolve(weights: str) -> Path:
    """Lokaler Ordner bleibt, HF-ID -> Snapshot-Ordner im HF-Cache."""
    if Path(weights).exists():
        return Path(weights)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=weights))


def main() -> int:
    if len(sys.argv) > 1:
        targets = [_resolve(p) for p in sys.argv[1:]]
    else:
        import config as C
        targets = [_resolve(C.MODELS[m]["weights"])
                   for m in ("M3_default_finetuned", "M4_spatial_config1", "M5_spatial_config2")]

    failed = []
    for model_dir in targets:
        print(f"\n=== {model_dir} ===")
        if not model_dir.exists():
            print("  FEHLT")
            failed.append(model_dir)
            continue
        ok1, msg1 = check_safetensors(model_dir)
        print(f"  [{'OK' if ok1 else 'FAIL'}] {msg1}")
        ok2, msg2 = (check_encoding(model_dir) if ok1 else (False, "uebersprungen (NaN-Scan failed)"))
        print(f"  [{'OK' if ok2 else 'FAIL'}] {msg2}")
        if not (ok1 and ok2):
            failed.append(model_dir)

    print()
    if failed:
        print(f"FAIL: {len(failed)}/{len(targets)} Modelle defekt: "
              + ", ".join(str(p) for p in failed))
        return 1
    print(f"OK: alle {len(targets)} Modelle intakt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
