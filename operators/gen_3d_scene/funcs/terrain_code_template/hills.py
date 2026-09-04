"""Rolling ground where what stands on it depends on the height.

The distribution is split by relief: towers take the ridges for the sightlines,
dwellings shelter in the hollows, and scrub fills the slopes between them. Read
against `plains`, this is what a terrain-aware layout buys.
"""
from __future__ import annotations

from .. import terrain_code_edit as te


def build(
    size: float = 76.0,
    relief: float = 4.0,
    towers: int = 4,
    huts: int = 7,
    scrub: int = 16,
    seed: int = 2,
) -> te.Scene:
    """Hills `relief` metres high across a `size` metre site."""
    terrain = te.hills(size, amplitude=relief, wavelength=34.0, tiles=26)
    reach = size * 0.78

    ridge = te.on_high_ground(
        te.scatter_spots(40, reach, seed=seed, min_gap=13.0),
        terrain, above=relief * 0.45,
    )[:towers]
    watch = te.place(
        "tower", "cylinder", ridge, size=(2.6, 7.5, 2.6), material="wall"
    )

    hollow = te.on_low_ground(
        te.scatter_spots(40, reach, seed=seed + 11, min_gap=10.0),
        terrain, below=-relief * 0.35,
    )[:huts]
    dwellings = te.place(
        "hut", "box", hollow,
        size=te.varied_sizes(len(hollow), (4.5, 3.0, 4.0),
                             spread=0.2, seed=seed),
        material="prop",
    )

    flanks = te.clear_of(
        te.scatter_spots(scrub, reach, seed=seed + 23, min_gap=6.0),
        watch + dwellings, margin=3.0,
    )
    bushes = te.place(
        "scrub", "sphere", flanks,
        size=te.varied_sizes(len(flanks), (2.2, 1.6, 2.2),
                             spread=0.3, seed=seed + 5),
        material="block",
    )

    return te.Scene(
        name="hills",
        terrain=terrain,
        props=watch + dwellings + bushes,
    )
