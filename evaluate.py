"""
Evaluation du comptage de fedisonar.py sur des clips CFC (Kenai channel) annotes.

Usage :
    python evaluate.py                   # les 69 clips
    python evaluate.py <clip> <clip>     # seulement ces clips
Pour chaque clip : poissons annotes / comptes / erreur, puis l'erreur absolue totale
et l'erreur relative (erreur absolue totale / nombre total de poissons annotes).
"""
import argparse, json
from pathlib import Path
from fedisonar import run

ANNOTATIONS = Path("data/annotations/kenai-channel")
METADATA = Path("data/metadata/kenai-channel.json")


def all_clips():
    """Noms des 69 clips, tries pour avoir toujours le meme ordre."""
    return sorted(m["clip_name"] for m in json.loads(METADATA.read_text()))


def annotated_count(clip):
    """Verite terrain : nombre de track_id distincts dans gt.txt (2e colonne)."""
    lines = (ANNOTATIONS / clip / "gt.txt").read_text().splitlines()
    return len({line.split(",")[1] for line in lines if line.strip()})


def evaluate(clips, verbose=True, **params):
    """Lance run() sur chaque clip ; renvoie (erreur absolue totale, erreur relative)."""
    total_true = total_abs = 0
    if verbose:
        print(f"{'clip':<70} {'annotes':>7} {'comptes':>7} {'erreur':>6}")
    for clip in clips:
        truth = annotated_count(clip)
        n = run(clip, save=False, **params)
        total_true += truth
        total_abs += abs(n - truth)
        if verbose:
            print(f"{clip:<70} {truth:>7} {n:>7} {n - truth:>+6}")
    rel = total_abs / total_true if total_true else 0.0
    if verbose:
        print(f"{len(clips)} clips | {total_true} poissons annotes | "
              f"erreur absolue totale : {total_abs} | erreur relative : {rel:.1%}")
    return total_abs, rel


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*", help="noms de clips (par defaut : les 69)")
    a = p.parse_args()
    evaluate(a.clips or all_clips())
