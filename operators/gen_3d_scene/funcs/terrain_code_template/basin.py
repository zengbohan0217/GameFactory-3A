"""Ground dishing down to a low centre, settled in rings up the slope.

Everything faces inward and the count grows with each ring outward, because a
bowl has one focus and increasing room around it. Water pools at the bottom
where a basin's drainage would put it.
"""
from __future__ import annotations

import math

from .. import terrain_code_edit as te

#: Footprint of one dwelling, and how far `varied_sizes` may grow it.
HOUSE = (4.0, 3.2, 3.4)
SPREAD = 0.18


def build(
    size: float = 72.0,
    depth: float = 8.0,
    terraces: int = 3,
    pool_radius: float = 11.0,
    seed: int = 3,
) -> te.Scene:
    """A basin `depth` metres deep with `terraces` rings of dwellings."""
    terrain = te.bowl(size, depth, tiles=30)

    water = [
        te.Prop("pool", "cylinder", (0.0, 0.0),
                (pool_radius * 2.0, 0.5, pool_radius * 2.0), material="water"),
    ]

    # Ring spacing comes from the dwelling's own diagonal, since each is turned
    # to face the centre and a turned box occupies its diagonal.
    reach = math.hypot(HOUSE[0], HOUSE[2]) * (1.0 + SPREAD)
    inner = pool_radius + reach
    outer = size / 2.0 - reach
    spread = max((outer - inner) / max(terraces - 1, 1), reach * 1.1)

    dwellings, posts = [], []
    for ring in range(terraces):
        radius = inner + ring * spread
        count = 6 + ring * 4
        spots = te.ring_spots(count, radius, start_degrees=ring * 12.0)
        dwellings += te.place(
            f"house-r{ring}", "box", spots,
            size=te.varied_sizes(count, HOUSE, spread=SPREAD, seed=seed + ring),
            material="wall", face_centre=True,
        )
        if ring == terraces - 1:
            posts += te.place(
                "beacon", "cylinder",
                te.ring_spots(4, radius + reach, start_degrees=45.0),
                size=(0.7, 5.0, 0.7), material="marker",
            )

    return te.Scene(
        name="basin",
        terrain=terrain,
        props=water + dwellings + posts,
    )
