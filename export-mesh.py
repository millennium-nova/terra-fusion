import argparse
from pathlib import Path

import imageio.v2 as imageio

from src.renderer import build_terrain_mesh


def main():
    parser = argparse.ArgumentParser(
        description="Convert a generated TerraFusion heightmap and texture to a mesh."
    )
    parser.add_argument("--heightmap", required=True, help="Path to heightmap TIFF.")
    parser.add_argument("--texture", required=True, help="Path to texture PNG.")
    parser.add_argument(
        "--output",
        default="terrain.ply",
        help="Output mesh path (.ply or .vtp; default: terrain.ply).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.suffix.lower() not in {".ply", ".vtp"}:
        parser.error("--output must use the .ply or .vtp extension")

    heightmap = imageio.imread(args.heightmap)
    texture = imageio.imread(args.texture)
    grid = build_terrain_mesh(texture, heightmap)
    try:
        surface = grid.extract_surface(algorithm=None)
    except TypeError:
        # PyVista versions before the algorithm option use this behavior directly.
        surface = grid.extract_surface()
    mesh = surface.triangulate()

    if output_path.suffix.lower() == ".ply":
        mesh.save(output_path, texture="RGB")
    else:
        mesh.save(output_path)

    print(
        f"Saved mesh: {output_path} "
        f"({mesh.n_points} vertices, {mesh.n_cells} triangles)"
    )


if __name__ == "__main__":
    main()
