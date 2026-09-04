"""Level open ground with scattered rock and thicket.

Nothing constrains movement and nothing repeats, so the distribution is a plain
random scatter at low density — the reference the other landforms depart from.
"""
from __future__ import annotations

from .. import terrain_code_edit as te


def build(
    size: float = 80.0,
    boulders: int = 14,
    thickets: int = 9,
    seed: int = 1,
) -> te.Scene:
    """A `size` metre plain carrying `boulders` rocks and `thickets` trees."""
    terrain = te.flat(size)

    rock_spots = te.scatter_spots(boulders, size * 0.85, seed=seed, min_gap=7.0)
    rocks = te.place(
        "boulder", "sphere", rock_spots,
        size=te.varied_sizes(len(rock_spots), (2.8, 2.2, 2.8),
                             spread=0.35, seed=seed),
        material="block",
    )

    tree_spots = te.clear_of(
        te.scatter_spots(thickets, size * 0.8, seed=seed + 7, min_gap=10.0),
        rocks, margin=3.5,
    )
    trunks = te.place(
        "trunk", "cylinder", tree_spots, size=(0.5, 3.4, 0.5), material="prop"
    )
    canopies = [
        te.Prop(f"canopy-{index:02d}", "sphere", spot, (4.6, 4.0, 4.6),
                material="block", sink=-3.4, group=f"tree-{index:02d}")
        for index, spot in enumerate(tree_spots)
    ]
    for index, trunk in enumerate(trunks):
        trunk.group = f"tree-{index:02d}"

    return te.Scene(
        name="plains",
        terrain=terrain,
        props=rocks + trunks + canopies,
    )
