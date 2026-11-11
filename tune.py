"""
Reglage des parametres de fedisonar.py sur le groupe de REGLAGE uniquement.
Le groupe de test n'est jamais charge ici : il sert seulement a la mesure finale (evaluate.py).

Usage : python tune.py
Recherche exhaustive sur une grille : methode de fond et son seuil, min_area, min_dx, min_len.
Critere : erreur absolue totale de comptage sur les 23 clips de reglage.
"""
import itertools
from multiprocessing import Pool
from evaluate import split, annotated_count
from fedisonar import load_gray, candidates, count, is_counted

# Detection (etape couteuse, calculee une fois par clip et par ligne)
DETECTIONS = [("mog2", dict(var_threshold=v)) for v in (16, 25, 40)] + \
             [("median", dict(diff_threshold=d)) for d in (10, 15, 20, 30, 40, 50)]
# min_area relatif (fraction de la surface de l'image) ...
MIN_AREA_REL = (0.00005, 0.0001, 0.0002, 0.0004, 0.0007, 0.0014)
# ... compare a min_area en pixels : toutes les strates ont ~137 px/m, un poisson y a donc
# la meme taille en pixels quelle que soit la taille de l'image
MIN_AREA_PX = (25, 50, 75, 100, 125, 150, 250, 400)
MIN_DX = (0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)          # fraction de la largeur ; 0 = pas de filtre
MIN_LEN = (3, 5, 8, 10, 15)                                 # images


def extract(job):
    """Taches en mouvement d'un clip pour une methode de detection (tourne dans un processus a part)."""
    clip, bg, params = job
    grays, _, _ = load_gray(clip)
    h, w = grays[0].shape
    return candidates(grays, bg, **params), w, h


def main():
    tune, _ = split()
    truth = [annotated_count(c) for c in tune]
    print(f"Reglage sur {len(tune)} clips ({sum(truth)} poissons annotes)")
    results = []
    with Pool(6) as pool:                                   # 6 clips en parallele (memoire : ~0,5 Go chacun)
        for bg, params in DETECTIONS:
            data = pool.map(extract, [(c, bg, params) for c in tune])
            areas = [("rel", a) for a in MIN_AREA_REL] + [("px", a) for a in MIN_AREA_PX]
            for mode, a in areas:
                # suivi une fois par clip ; min_len et min_dx ne font ensuite que filtrer les pistes
                tracks = [count(cands, w, h, a * w * h if mode == "rel" else a)[0] for cands, w, h in data]
                for dx, ln in itertools.product(MIN_DX, MIN_LEN):
                    err = sum(abs(sum(is_counted(t, w, ln, dx) for t in trs) - n)
                              for trs, (_, w, h), n in zip(tracks, data, truth))
                    results.append((err, bg, params, mode, a, dx, ln))
            best = min((r for r in results if r[1] == bg and r[2] == params), key=lambda r: r[0])
            print(f"  {bg} {params} : meilleure erreur {best[0]}")
    results.sort(key=lambda r: r[0])
    for mode in ("rel", "px"):
        print(f"\n10 meilleurs reglages, min_area {'relatif' if mode == 'rel' else 'en pixels'} :")
        print(f"{'erreur':>6} {'fond':<7} {'seuil':<22} {'min_area':>8} {'min_dx':>6} {'min_len':>7}")
        for err, bg, params, m, a, dx, ln in [r for r in results if r[3] == mode][:10]:
            print(f"{err:>6} {bg:<7} {str(params):<22} {a:>8} {dx:>6} {ln:>7}")


if __name__ == "__main__":
    main()
