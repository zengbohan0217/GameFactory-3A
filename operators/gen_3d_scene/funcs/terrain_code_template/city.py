"""Level ground built out as a regular street grid.

The same flat terrain as `plains`, filled the opposite way: an orthogonal
lattice on a fixed pitch, so streets run straight through and every gap is the
same width. Height carries the variation instead of position, tapering from a
tall core to low outskirts.
"""
from __future__ import annotations

from .. import terrain_code_edit as te


def build(
    blocks: int = 5,
    block_size: float = 13.0,
    street: float = 9.0,
    core_height: float = 15.0,
    edge_height: float = 4.0,
) -> te.Scene:
    """A `blocks` by `blocks` grid of buildings separated by streets."""
    pitch = block_size + street
    size = blocks * pitch + street * 2.0
    terrain = te.flat(size)

    spots = te.grid_spots(blocks, blocks, pitch)
    middle = (blocks - 1) / 2.0

    towers = []
    for index, spot in enumerate(spots):
        row, column = divmod(index, blocks)
        # The middle block is left open for the plaza, when there is one.
        if row == middle and column == middle:
            continue
        # Distance from the centre block, as a fraction of the way to a corner.
        reach = max(abs(row - middle), abs(column - middle)) / max(middle, 1e-6)
        height = core_height + (edge_height - core_height) * reach
        towers.append(
            te.Prop(f"block-{row}-{column}", "box", spot,
                    (block_size, height, block_size), material="wall")
        )

    lamps = te.place(
        "lamp", "cylinder",
        te.grid_spots(blocks - 1, blocks - 1, pitch),
        size=(0.4, 3.6, 0.4), material="marker",
    )

    plaza = [
        te.Prop("plaza", "box", (0.0, 0.0),
                (block_size * 0.9, 0.15, block_size * 0.9), material="block"),
    ]

    return te.Scene(
        name="city",
        terrain=terrain,
        props=towers + lamps + plaza,
    )
