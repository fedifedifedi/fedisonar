"""
FediSonar - prototype de Fedi Limem : detection, suivi et comptage d'objets en mouvement dans une video sonar.
Approche classique (sans entrainement) servant de baseline avant un modele deep learning.

Usage :
    python fedisonar.py video.mp4                # video reelle
    python fedisonar.py <nom_du_clip>            # clip CFC : images data/channel/<nom_du_clip>_N.jpg
    python fedisonar.py --demo                   # genere une video synthetique de test
    option --fond mog2                           # fond appris au fil de l'eau au lieu de la mediane du clip
Sorties : <nom>_annotee.mp4 (boites + trajectoires + compteurs) et <nom>_pistes.csv
          (dans outputs/ pour un clip CFC)
"""
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np


def make_demo(path="fedisonar_demo.mp4", n=300, w=480, h=640, seed=0):
    """Video synthetique type sonar : bruit de speckle + formes allongees qui traversent."""
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h), False)
    # (frame de depart, y, vitesse en px/frame : >0 = gauche->droite, longueur)
    fish = [(10, 150, 4, 60), (60, 320, -5, 80), (120, 480, 3, 50), (170, 230, 6, 70), (200, 400, -4, 65)]
    static = (rng.random((h, w)) * 40).astype(np.float32)          # fond fixe (berge, fond de riviere)
    cv2.rectangle(static, (0, h - 60), (w, h), 90, -1)
    for t in range(n):
        f = static + rng.rayleigh(18, (h, w)).astype(np.float32)   # speckle acoustique
        for t0, y, v, L in fish:
            if t >= t0:
                x = (-L if v > 0 else w + L) + v * (t - t0)
                yy = int(y + 12 * np.sin((t - t0) / 6))            # ondulation de la nage
                cv2.ellipse(f, (int(x), yy), (L // 2, 7), 0, 0, 360, 210, -1)
        out.write(np.clip(cv2.GaussianBlur(f, (5, 5), 0), 0, 255).astype(np.uint8))
    out.release()
    return path, len(fish)


class Tracker:
    """Suivi par association au plus proche voisin (centroides)."""
    def __init__(self, max_dist=60, max_missed=8):
        self.max_dist, self.max_missed = max_dist, max_missed
        self.tracks, self.done, self.next_id = {}, [], 1

    def update(self, dets, frame_idx):
        free = set(range(len(dets)))
        for tid, tr in list(self.tracks.items()):
            best, bd = None, self.max_dist
            for i in free:
                d = np.hypot(dets[i][0] - tr["pts"][-1][0], dets[i][1] - tr["pts"][-1][1])
                if d < bd:
                    best, bd = i, d
            if best is None:
                tr["missed"] += 1
                if tr["missed"] > self.max_missed:
                    self.done.append(self.tracks.pop(tid))
            else:
                free.discard(best)
                tr["pts"].append(dets[best][:2]); tr["boxes"].append(dets[best][2])
                tr["frames"].append(frame_idx); tr["missed"] = 0
        for i in free:
            # historique complet (position, boite, numero d'image) : sert a dessiner la video apres coup
            self.tracks[self.next_id] = dict(id=self.next_id, pts=[dets[i][:2]], boxes=[dets[i][2]],
                                             frames=[frame_idx], missed=0)
            self.next_id += 1

    def all(self):
        return self.done + list(self.tracks.values())


def clip_files(name, folder="data/channel"):
    """Images d'un clip CFC (<nom>_N.jpg), triees par N en NUMERIQUE : sinon _10 passerait avant _2."""
    files = [p for p in Path(folder).glob(f"{name}_*.jpg") if p.stem.rsplit("_", 1)[0] == name]
    return sorted(files, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def load_gray(video):
    """Toutes les images (video ou clip CFC) en niveaux de gris, la cadence et le prefixe des sorties.
    Tout est garde en memoire : le fond median a besoin du clip entier (200 images, 150 Mo au plus)."""
    if Path(video).is_file():                                                         # fichier video
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            sys.exit(f"Impossible d'ouvrir {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        grays = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        return grays, fps, Path(video).with_suffix("")
    files = clip_files(video)                                                         # sinon : nom d'un clip CFC
    if not files:
        sys.exit(f"Ni video ni clip trouve pour {video}")
    # CFC channel : 8 images/s (metadonnees) ; les JPG bruts sont deja en niveaux de gris
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files], 8, Path("outputs") / video


def candidates(grays, bg="median", var_threshold=40, diff_threshold=30):
    """Taches en mouvement de chaque image : liste de (cx, cy, boite, aire), AVANT le filtre d'aire.
    C'est l'etape couteuse ; la separer permet de la calculer une seule fois pendant le reglage.
    bg="mog2"   : fond appris au fil des images (marcherait en direct), 20 images d'apprentissage.
    bg="median" : fond = mediane de toutes les images du clip. Possible car le clip est enregistre :
                  un poisson ne reste pas au meme endroit la plupart du temps, donc la mediane
                  donne le fond sans poisson, et on detecte des la premiere image."""
    blurred = [cv2.GaussianBlur(g, (7, 7), 0) for g in grays]                        # attenue le speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if bg == "median":
        background = np.median(np.stack(blurred), axis=0).astype(np.uint8)
    else:
        mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=var_threshold, detectShadows=False)
    result = []
    for k, g in enumerate(blurred):
        if bg == "median":
            # un poisson renvoie plus d'echo que le fond : on garde seulement ce qui est PLUS clair
            # (cv2.subtract ramene a 0 les differences negatives)
            mask = cv2.threshold(cv2.subtract(g, background), diff_threshold, 255, cv2.THRESH_BINARY)[1]
        else:
            mask = mog.apply(g)                                                       # fond appris -> objets mobiles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cands = []
        if bg == "median" or k > 20:                                                  # MOG2 : laisse le fond se stabiliser
            for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                x, y, bw, bh = cv2.boundingRect(c)
                cands.append((x + bw / 2, y + bh / 2, (x, y, bw, bh), cv2.contourArea(c)))
        result.append(cands)
    return result


def count(cands, w, h, min_area=125, min_len=10, min_dx=0.02):
    """Suivi puis comptage. Valeurs par defaut reglees sur le groupe de reglage CFC (tune.py).
    min_area en PIXELS : toutes les strates CFC ont ~137 px/m, un poisson y a donc la meme taille
    en pixels, que l'image fasse 216x485 ou 707x1098 (une aire relative a l'image faisait pire).
    min_dx en fraction de la LARGEUR : un poisson doit traverser une part du champ, quelle que
    soit sa largeur. Renvoie (toutes les pistes, pistes comptees)."""
    trk = Tracker()
    for k, frame_cands in enumerate(cands):
        trk.update([(cx, cy, box) for cx, cy, box, area in frame_cands if area >= min_area], k)
    tracks = trk.all()
    return tracks, [t for t in tracks if is_counted(t, w, min_len, min_dx)]


def is_counted(t, w, min_len, min_dx):
    """Une piste compte si elle dure assez et traverse vraiment le champ (min_dx : fraction de la largeur w)."""
    return len(t["pts"]) >= min_len and abs(t["pts"][-1][0] - t["pts"][0][0]) >= min_dx * w


def going_right(t):
    return t["pts"][-1][0] > t["pts"][0][0]


def write_video(grays, tracks, counted, path, fps):
    """Video annotee, dessinee apres coup grace a l'historique des pistes.
    Vert = piste comptee, rouge = piste rejetee (trop courte ou pas assez de deplacement)."""
    h, w = grays[0].shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    counted_ids = {t["id"] for t in counted}
    for k, g in enumerate(grays):
        frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            if k in t["frames"]:                                                      # piste detectee sur cette image
                i = t["frames"].index(k)
                x, y, bw, bh = t["boxes"][i]
                color = (0, 255, 0) if t["id"] in counted_ids else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, f"#{t['id']}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.polylines(frame, [np.int32(t["pts"][:i + 1])], False, (0, 200, 255), 1)
        done = [t for t in counted if t["frames"][-1] <= k]                           # compte quand la piste se termine
        lr = sum(going_right(t) for t in done)
        cv2.putText(frame, f"G->D: {lr}   D->G: {len(done) - lr}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def run(video, min_area=125, min_len=10, min_dx=0.02, bg="median", save=True):
    # save=False : compte seulement, sans ecrire video ni CSV (evaluation rapide sur beaucoup de clips)
    grays, fps, stem = load_gray(video)
    h, w = grays[0].shape
    tracks, counted = count(candidates(grays, bg), w, h, min_area, min_len, min_dx)
    if not save:
        return len(counted)
    stem.parent.mkdir(exist_ok=True)
    write_video(grays, tracks, counted, f"{stem}_annotee.mp4", fps)
    with open(f"{stem}_pistes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "frame_debut", "frame_fin", "duree_s", "sens", "largeur_derniere_boite_px"])
        for t in counted:
            first, last = t["frames"][0], t["frames"][-1]
            sens = "gauche->droite" if going_right(t) else "droite->gauche"
            wr.writerow([t["id"], first, last, round((last - first) / fps, 2), sens, t["boxes"][-1][2]])
    lr = sum(going_right(t) for t in counted)
    print(f"{len(grays)} frames | {len(counted)} objets comptes (gauche->droite: {lr}, droite->gauche: {len(counted) - lr})")
    print(f"Sorties : {stem}_annotee.mp4, {stem}_pistes.csv")
    return len(counted)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?")
    p.add_argument("--demo", action="store_true", help="genere et traite une video synthetique")
    p.add_argument("--min-area", type=int, default=125, help="aire minimale d'une tache, en pixels")
    p.add_argument("--fond", choices=["mog2", "median"], default="median", help="methode de fond")
    a = p.parse_args()
    if a.demo:
        path, truth = make_demo()
        n = run(path, a.min_area, bg=a.fond)
        print(f"Verite terrain de la demo : {truth} objets -> detectes : {n}")
    elif a.video:
        run(a.video, a.min_area, bg=a.fond)
    else:
        p.print_help()
