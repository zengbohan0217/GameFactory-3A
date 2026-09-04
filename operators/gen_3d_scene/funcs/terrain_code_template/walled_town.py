"""A raised plateau ringed by a wall, built outward from its centre.

The mound supplies the defensive height and the wall follows its flat rim, so
the enclosure comes from the landform rather than being dropped onto it.
Inside, buildings sit on radial spokes: a boundary with one focus produces
streets that run to it.
"""
from __future__ import annotations

import math

from .. import terrain_code_edit as te

#: Footprint of one house. Width runs along the spoke, depth across it.
HOUSE = (4.2, 3.6)

#: Side of the square keep at the centre.
KEEP = 6.0


def build(
    size: float = 80.0,
    rise: float = 5.0,
    wall_radius: float = 21.0,
    segments: int = 20,
    spokes: int = 6,
    depth: int = 3,
) -> te.Scene:
    """A town on a `rise` metre mound behind a wall `wall_radius` out.

    `depth` is the most rings of houses to lay along each spoke; fewer are
    placed when the wall is too close to hold them all.
    """
    terrain = te.mound(size, rise, flat_radius=wall_radius + 3.0, tiles=28)

    chord = 2.0 * math.pi * wall_radius / segments
    wall_thickness = 1.0

    wall = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        wall.append(
            te.Prop(
                f"wall-{index:02d}", "box",
                (wall_radius * math.cos(angle), wall_radius * math.sin(angle)),
                (chord * 1.08, 4.2, wall_thickness),
                yaw=-math.degrees(angle + math.pi / 2.0),
                material="wall", group="rampart",
            )
        )

    # Towers stand in the wall rather than behind it, so they share its group.
    towers = [
        te.Prop(f"gate-tower-{index:02d}", "cylinder", spot,
                (2.6, 6.0, 2.6), material="marker", group="rampart")
        for index, spot in enumerate(
            te.ring_spots(4, wall_radius, start_degrees=45.0)
        )
    ]

    # Spoke spacing follows the house: its width runs along the spoke, so that
    # sets the gap between rings, and the innermost ring has to clear the keep.
    along = HOUSE[0]
    first = KEEP * math.sqrt(2.0) / 2.0 + along / 2.0 + 0.6
    step = along * 1.15
    limit = wall_radius - wall_thickness / 2.0 - along / 2.0

    houses = []
    for spoke in range(spokes):
        angle = 2.0 * math.pi * spoke / spokes
        for ring in range(depth):
            radius = first + ring * step
            if radius > limit:
                break
            houses.append(
                te.Prop(
                    f"house-{spoke}-{ring}", "box",
                    (radius * math.cos(angle), radius * math.sin(angle)),
                    (HOUSE[0], 3.4 + 0.4 * (depth - ring), HOUSE[1]),
                    yaw=-math.degrees(angle), material="prop",
                )
            )

    keep = [te.Prop("keep", "box", (0.0, 0.0), (KEEP, 8.0, KEEP), material="block")]

    return te.Scene(
        name="walled_town",
        terrain=terrain,
        props=wall + towers + houses + keep,
    )
