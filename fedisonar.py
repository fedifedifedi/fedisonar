"""
FediSonar - prototype de Fedi Limem : detection, suivi et comptage d'objets en mouvement dans une video sonar.
Approche classique (sans entrainement) servant de baseline avant un modele deep learning.

Usage :
    python fedisonar.py video.mp4                # video reelle
    python fedisonar.py <nom_du_clip>            # clip CFC : images data/channel/<nom_du_clip>_N.jpg
    python fedisonar.py --demo                   # genere une video synthetique de test
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
                tr["pts"].append(dets[best][:2]); tr["box"] = dets[best][2]
                tr["last"] = frame_idx; tr["missed"] = 0
        for i in free:
            self.tracks[self.next_id] = dict(id=self.next_id, pts=[dets[i][:2]], box=dets[i][2],
                                             first=frame_idx, last=frame_idx, missed=0)
            self.next_id += 1

    def all(self):
        return self.done + list(self.tracks.values())


def clip_files(name, folder="data/channel"):
    """Images d'un clip CFC (<nom>_N.jpg), triees par N en NUMERIQUE : sinon _10 passerait avant _2."""
    files = [p for p in Path(folder).glob(f"{name}_*.jpg") if p.stem.rsplit("_", 1)[0] == name]
    return sorted(files, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def video_frames(cap):
    """Lit une video image par image, pour la traiter comme une liste d'images."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def run(video, min_area=150, min_len=10, min_dx=80):
    cap = None
    if Path(video).is_file():                                                         # fichier video
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            sys.exit(f"Impossible d'ouvrir {video}")
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        stem = Path(video).with_suffix("")
        frames = video_frames(cap)
    else:                                                                             # sinon : nom d'un clip CFC
        files = clip_files(video)
        if not files:
            sys.exit(f"Ni video ni clip trouve pour {video}")
        h, w = cv2.imread(str(files[0])).shape[:2]
        fps = 8                                                                       # CFC channel : 8 images/s (metadonnees)
        Path("outputs").mkdir(exist_ok=True)
        stem = Path("outputs") / video
        frames = (cv2.imread(str(p)) for p in files)                                  # lues une par une, pas toutes en memoire
    out = cv2.VideoWriter(f"{stem}_annotee.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trk, k = Tracker(), 0

    def valid(tr):  # une piste compte si elle dure assez et traverse vraiment le champ
        return len(tr["pts"]) >= min_len and abs(tr["pts"][-1][0] - tr["pts"][0][0]) >= min_dx

    for frame in frames:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)   # attenue le speckle
        mask = bg.apply(gray)                                                         # fond appris -> objets mobiles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dets = []
        if k > 20:                                                                    # laisse le fond se stabiliser
            for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(c) >= min_area:
                    x, y, bw, bh = cv2.boundingRect(c)
                    dets.append((x + bw / 2, y + bh / 2, (x, y, bw, bh)))
        trk.update(dets, k)
        for tr in trk.tracks.values():
            if tr["missed"] == 0:
                x, y, bw, bh = tr["box"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"#{tr['id']}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.polylines(frame, [np.int32(tr["pts"])], False, (0, 200, 255), 1)
        v = [t for t in trk.all() if valid(t)]
        lr = sum(t["pts"][-1][0] > t["pts"][0][0] for t in v)
        cv2.putText(frame, f"G->D: {lr}   D->G: {len(v) - lr}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame); k += 1
    if cap:
        cap.release()
    out.release()

    v = [t for t in trk.all() if valid(t)]
    with open(f"{stem}_pistes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "frame_debut", "frame_fin", "duree_s", "sens", "largeur_derniere_boite_px"])
        for t in v:
            sens = "gauche->droite" if t["pts"][-1][0] > t["pts"][0][0] else "droite->gauche"
            wr.writerow([t["id"], t["first"], t["last"], round((t["last"] - t["first"]) / fps, 2), sens, t["box"][2]])
    lr = sum(t["pts"][-1][0] > t["pts"][0][0] for t in v)
    print(f"{k} frames | {len(v)} objets comptes (gauche->droite: {lr}, droite->gauche: {len(v) - lr})")
    print(f"Sorties : {stem}_annotee.mp4, {stem}_pistes.csv")
    return len(v)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?")
    p.add_argument("--demo", action="store_true", help="genere et traite une video synthetique")
    p.add_argument("--min-area", type=int, default=150)
    a = p.parse_args()
    if a.demo:
        path, truth = make_demo()
        n = run(path, a.min_area)
        print(f"Verite terrain de la demo : {truth} objets -> detectes : {n}")
    elif a.video:
        run(a.video, a.min_area)
    else:
        p.print_help()
