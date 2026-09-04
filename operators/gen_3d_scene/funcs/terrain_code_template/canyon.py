"""A level channel between rising walls, worked along its length.

The walls leave one axis to move along, so the distribution is a chain rather
than a field: waypoints down the floor, and debris caught on the steep flanks
where it would have fallen.
"""
from __future__ import annotations

from .. import terrain_code_edit as te


def build(
    size: float = 78.0,
    depth: float = 9.0,
    floor_width: float = 16.0,
    waypoints: int = 7,
    debris: int = 18,
    seed: int = 4,
) -> te.Scene:
    """A canyon `floor_width` metres across with walls `depth` metres high."""
    terrain = te.canyon(size, depth, floor_width, tiles=30)

    run = size * 0.8
    spacing = run / max(waypoints - 1, 1)
    centreline = te.line_spots(waypoints, spacing, angle_degrees=90.0)

    markers = te.place(
        "marker", "cylinder", centreline, size=(0.6, 2.8, 0.6), material="marker"
    )

    gates = []
    for index, (x, z) in enumerate(centreline):
        if index % 2:
            continue
        offset = floor_width / 2.0 - 1.5
        gates += [
            te.Prop(f"post-w-{index:02d}", "box", (x - offset, z),
                    (0.8, 4.0, 0.8), material="wall"),
            te.Prop(f"post-e-{index:02d}", "box", (x + offset, z),
                    (0.8, 4.0, 0.8), material="wall"),
        ]

    fallen = te.on_slope(
        te.scatter_spots(debris * 3, size * 0.85, seed=seed, min_gap=5.0),
        terrain, steeper_than=0.25,
    )[:debris]
    rocks = te.place(
        "rock", "sphere", fallen,
        size=te.varied_sizes(len(fallen), (2.0, 1.6, 2.0),
                             spread=0.4, seed=seed + 9),
        material="block",
    )

    return te.Scene(
        name="canyon",
        terrain=terrain,
        props=markers + gates + rocks,
    )
