"""
Tableau final avant/apres reglage, mesure sur le groupe de TEST (jamais utilise pour regler).

Usage : python report.py
- Seuils retenus convertis en unites physiques (cm², m), avec la resolution de chaque clip
  tiree des metadonnees (largeur/hauteur en px et etendue en metres).
- Par strate puis au total : poissons annotes, comptes, erreur signee (>0 : surcomptage,
  <0 : sous-comptage), erreur absolue (somme des |erreurs| par clip) et erreur relative.
"""
import json
from multiprocessing import Pool
from evaluate import split, annotated_count, METADATA
from fedisonar import load_gray, candidates, count

# min_area en pixels ; min_dx en pixels ("px") ou en fraction de la largeur ("rel")
AVANT = dict(bg="mog2", det=dict(var_threshold=40), min_area=150, min_dx=("px", 80), min_len=10)
APRES = dict(bg="median", det=dict(diff_threshold=30), min_area=125, min_dx=("rel", 0.02), min_len=10)


def stratum(clip):
    return clip.split("_")[2]                                  # ex. "Stratum1"


def resolution(meta):
    """Pixels par metre en x (largeur) et en y (hauteur) d'un clip."""
    return (meta["width"] / (meta["x_meter_stop"] - meta["x_meter_start"]),
            meta["height"] / (meta["y_meter_start"] - meta["y_meter_stop"]))


def dx_px(cfg, w):
    mode, value = cfg["min_dx"]
    return value if mode == "px" else value * w


def counted(cands, w, h, cfg):
    return len(count(cands, w, h, cfg["min_area"], cfg["min_len"], dx_px(cfg, w) / w)[1])


def run_clip(clip):
    """Comptes avant et apres pour un clip (tourne dans un processus a part)."""
    grays, _, _ = load_gray(clip)
    h, w = grays[0].shape
    return [counted(candidates(grays, cfg["bg"], **cfg["det"]), w, h, cfg) for cfg in (AVANT, APRES)]


def physical_units(clips, metas):
    print("Seuils retenus en unites physiques (min - max sur les clips de chaque strate) :")
    print(f"{'strate':<9} {'images (px)':<13} {'px/m':>9} | {'min_area':>15} | {'min_dx':>15}")
    for s in sorted({stratum(c) for c in clips}):
        rows = []
        for c in clips:
            if stratum(c) == s:
                m = metas[c]
                rx, ry = resolution(m)
                area_cm2 = APRES["min_area"] / (rx * ry) * 1e4                # px² -> m² -> cm²
                dx_m = dx_px(APRES, m["width"]) / rx
                rows.append((m["width"], m["height"], rx, area_cm2, dx_m))
        lo, hi = lambda i: min(r[i] for r in rows), lambda i: max(r[i] for r in rows)
        print(f"{s:<9} {f'{lo(0)}x{lo(1)}':<13} {lo(2):>4.0f}-{hi(2):<4.0f} | "
              f"{lo(3):>5.1f}-{hi(3):<4.1f} cm² | {lo(4):>5.2f}-{hi(4):<4.2f} m")
    print(f"(min_area = {APRES['min_area']} px, min_dx = {APRES['min_dx'][1]:.0%} de la largeur)\n")


def table(clips, truth, results):
    print(f"Groupe de test : avant = {AVANT}\n                 apres = {APRES}\n")
    print(f"{'':<10} {'':>5} {'':>7} | {'AVANT':^31} | {'APRES':^31}")
    print(f"{'strate':<10} {'clips':>5} {'annotes':>7} | "
          + " | ".join(f"{'comptes':>7} {'signee':>6} {'absolue':>7} {'rel.':>6}" for _ in range(2)))
    groups = sorted({stratum(c) for c in clips}) + ["TOTAL"]
    for g in groups:
        idx = [i for i, c in enumerate(clips) if g == "TOTAL" or stratum(c) == g]
        n_true = sum(truth[i] for i in idx)
        cells = []
        for k in range(2):                                     # 0 = avant, 1 = apres
            n = sum(results[i][k] for i in idx)
            abs_err = sum(abs(results[i][k] - truth[i]) for i in idx)
            cells.append(f"{n:>7} {n - n_true:>+6} {abs_err:>7} {abs_err / n_true:>6.1%}")
        print(f"{g:<10} {len(idx):>5} {n_true:>7} | " + " | ".join(cells))


def main():
    _, test = split()
    metas = {m["clip_name"]: m for m in json.loads(METADATA.read_text())}
    physical_units(test, metas)
    truth = [annotated_count(c) for c in test]
    with Pool(6) as pool:
        results = pool.map(run_clip, test)
    table(test, truth, results)


if __name__ == "__main__":
    main()
