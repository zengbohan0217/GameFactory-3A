"""
operators/gen_3d_scene/funcs/terrain_code_template/

Greybox scenes, one module per kind of terrain.

Modules are named for the landform, not for a level: `basin` is a bowl with
whatever a bowl implies, and any number of games can be set in one. Each pairs
a terrain shape with the distribution that shape produces, because the two are
not independent — a settlement in a basin rings the slope, the same settlement
on a ridge follows the high ground.

| module        | terrain          | distribution                       |
|---------------|------------------|------------------------------------|
| `plains`      | level, wide      | sparse scatter, no structure       |
| `hills`       | rolling          | split by height: ridge vs hollow   |
| `basin`       | dished, low centre | concentric rings up the slope    |
| `canyon`      | channel, walls   | linear along the floor             |
| `walled_town` | raised plateau   | perimeter ring, radial inside      |
| `city`        | level            | dense orthogonal grid              |

A template is a function returning a `Scene`. It picks a terrain, a layout and
a set of sizes from `terrain_code_edit` and does nothing else — no file writing, no
asset fetching — so a caller can adjust the result before anything is built.

Every template takes keyword arguments for the handful of numbers worth
changing and leaves the rest as constants, so calling one with no arguments
gives a scene that already passes `check_scene`.

Usage:
    from operators.gen_3d_scene.funcs.terrain_code_template import basin

    scene = basin.build(size=70.0)
"""
from __future__ import annotations

from . import basin, canyon, city, hills, plains, walled_town

#: Templates by landform, so a task can pick one from a string.
TEMPLATES = {
    "plains": plains.build,
    "hills": hills.build,
    "basin": basin.build,
    "canyon": canyon.build,
    "walled_town": walled_town.build,
    "city": city.build,
}

__all__ = [
    "TEMPLATES",
    "basin",
    "canyon",
    "city",
    "hills",
    "plains",
    "walled_town",
]
