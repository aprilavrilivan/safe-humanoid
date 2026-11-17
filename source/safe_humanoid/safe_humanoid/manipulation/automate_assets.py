"""Inspect AutoMate OBJ geometry before choosing representative assembly IDs."""

import json
import math
from pathlib import Path


def inspect_assembly(directory: Path) -> dict:
    """Return reproducible mesh descriptors; do not infer task difficulty."""
    directory = Path(directory)
    result = {"assembly_id": directory.name, "parts": {}}
    for part in ("plug", "socket"):
        obj = directory / f"{part}.obj"
        if not obj.is_file():
            raise FileNotFoundError(obj)
        lows = [math.inf] * 3
        highs = [-math.inf] * 3
        vertices = 0
        faces = 0
        with obj.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("v "):
                    xyz = [float(x) for x in line.split()[1:4]]
                    if len(xyz) != 3:
                        raise ValueError(f"malformed vertex in {obj}")
                    lows = [min(a, b) for a, b in zip(lows, xyz)]
                    highs = [max(a, b) for a, b in zip(highs, xyz)]
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
        if not vertices:
            raise ValueError(f"no vertices in {obj}")
        result["parts"][part] = {"vertices": vertices, "faces": faces,
                                  "bbox_extent": [round(hi - lo, 7) for lo, hi in zip(lows, highs)]}
    return result


def select_diverse(root: Path) -> list[dict]:
    """Select three distinct bounding-box/mesh profiles from available assets.

    OBJ coordinates and face counts are only geometry proxies; a human must
    inspect screenshots and asset metadata before calling them representative.
    """
    assets = []
    for directory in sorted(Path(root).iterdir()):
        if directory.is_dir() and directory.name.isdigit() and len(directory.name) == 5:
            try:
                assets.append(inspect_assembly(directory))
            except FileNotFoundError:
                continue
    if len(assets) < 3:
        raise ValueError("need at least three local AutoMate assemblies with plug.obj and socket.obj")
    def features(asset):
        plug = asset["parts"]["plug"]
        socket = asset["parts"]["socket"]
        return [math.log1p(plug["faces"]), math.log1p(socket["faces"]),
                *(math.log1p(1000*x) for x in plug["bbox_extent"]),
                *(math.log1p(1000*x) for x in socket["bbox_extent"])]
    vectors = [features(asset) for asset in assets]
    span = [max(v[i] for v in vectors) - min(v[i] for v in vectors) or 1 for i in range(len(vectors[0]))]
    chosen = [0]
    while len(chosen) < 3:
        def distance(index):
            return min(sum(((vectors[index][k] - vectors[j][k]) / span[k]) ** 2 for k in range(len(span)))
                       for j in chosen)
        chosen.append(max((i for i in range(len(assets)) if i not in chosen), key=lambda i: (distance(i), -i)))
    return [assets[i] for i in chosen]


def write_selection(root: Path, destination: Path) -> list[dict]:
    selection = select_diverse(root)
    Path(destination).write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection
