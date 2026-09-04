"""
operators/gen_3d_scene/funcs/terrain_code_edit.py

Build a scene from code in two stages: a greybox first, then detail.

Four concerns stay separate, and each is an ordinary value:

    terrain    what the ground is
    layout     where things stand on it
    sizing     how big each one is
    materials  what colour or texture it reads as

A template picks one of each; a caller overrides any single one without
touching the rest.

Stage 1 writes a greybox — primitives with flat colours — and `check_scene`
reports whether the arrangement is usable before any detail is paid for.
Stage 2 replaces individual props with generated meshes and stages ground
textures into a game project.

Usage:
    from operators.gen_3d_scene.funcs import terrain_code_edit as te

    scene = te.Scene(
        name="clearing",
        terrain=te.flat(48.0),
        props=te.place("pillar", "cylinder",
                       te.ring_spots(6, 12.0), size=(1.2, 5.0, 1.2)),
    )
    print(te.check_scene(scene))
    te.write_scene(scene, "greybox.glb")
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

Spot = tuple[float, float]
Vec3 = tuple[float, float, float]

#: Spec dialect written by `models/common/glb_writer.py`.
UNITS = "metres"

#: Height of a standing human, used as the scale reference in `check_scene`.
HUMAN_HEIGHT = 1.8


# ── materials ────────────────────────────────────────────────────────────────

#: Greybox palette. Distinct enough to tell surfaces apart, flat enough that
#: nobody mistakes the result for finished art.
GREYBOX_MATERIALS: dict[str, dict[str, Any]] = {
    "ground": {"baseColor": [0.55, 0.56, 0.52, 1.0], "roughness": 0.95},
    "block": {"baseColor": [0.72, 0.72, 0.74, 1.0], "roughness": 0.80},
    "wall": {"baseColor": [0.62, 0.60, 0.58, 1.0], "roughness": 0.85},
    "prop": {"baseColor": [0.70, 0.58, 0.42, 1.0], "roughness": 0.75},
    "water": {"baseColor": [0.30, 0.48, 0.62, 0.65], "roughness": 0.15},
    "marker": {"baseColor": [0.85, 0.35, 0.30, 1.0], "roughness": 0.60},
}

#: Material name -> asset id in `scene_assets.SCENE_ASSETS`. Read by
#: `stage_scene_textures` to fetch the files a finished scene needs.
MATERIAL_TEXTURES: dict[str, str] = {
    "ground": "tex_grass_ground",
    "wall": "tex_brick_wall",
    "block": "tex_grain_noise",
    "water": "tex_water_normals",
}


# ── terrain ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Terrain:
    """Ground the props stand on.

    A `height` of None is a flat slab written as one box. Any other callable
    is sampled per tile, so `tiles` controls how finely the surface is
    stepped.
    """

    size: float = 48.0
    thickness: float = 0.4
    tiles: int = 1
    height: Callable[[float, float], float] | None = None
    material: str = "ground"


def flat(size: float = 48.0, material: str = "ground") -> Terrain:
    """Level ground."""
    return Terrain(size=size, material=material)


def hills(
    size: float = 48.0,
    amplitude: float = 2.0,
    wavelength: float = 24.0,
    tiles: int = 24,
    material: str = "ground",
) -> Terrain:
    """Rolling ground from two crossed sine waves."""

    def height(x: float, z: float) -> float:
        k = 2.0 * math.pi / max(wavelength, 1e-6)
        return amplitude * 0.5 * (math.sin(k * x) + math.cos(k * z))

    return Terrain(size=size, tiles=tiles, height=height, material=material)


def slope(
    size: float = 48.0,
    rise: float = 6.0,
    axis: str = "z",
    tiles: int = 16,
    material: str = "ground",
) -> Terrain:
    """Ground climbing steadily along one axis."""

    def height(x: float, z: float) -> float:
        along = z if axis == "z" else x
        return rise * (along / max(size, 1e-6) + 0.5)

    return Terrain(size=size, tiles=tiles, height=height, material=material)


def bowl(
    size: float = 64.0,
    depth: float = 7.0,
    tiles: int = 28,
    material: str = "ground",
) -> Terrain:
    """Ground dishing down to a low centre, level at the rim."""
    half = max(size / 2.0, 1e-6)

    def height(x: float, z: float) -> float:
        radial = min(math.hypot(x, z) / half, 1.0)
        return depth * (radial * radial - 1.0)

    return Terrain(size=size, tiles=tiles, height=height, material=material)


def mound(
    size: float = 64.0,
    rise: float = 6.0,
    flat_radius: float = 14.0,
    tiles: int = 28,
    material: str = "ground",
) -> Terrain:
    """A raised plateau with a level top and sloping flanks."""
    half = max(size / 2.0, 1e-6)
    run = max(half - flat_radius, 1e-6)

    def height(x: float, z: float) -> float:
        radial = math.hypot(x, z)
        if radial <= flat_radius:
            return rise
        return rise * max(0.0, 1.0 - (radial - flat_radius) / run)

    return Terrain(size=size, tiles=tiles, height=height, material=material)


def canyon(
    size: float = 72.0,
    depth: float = 9.0,
    floor_width: float = 16.0,
    tiles: int = 30,
    material: str = "ground",
) -> Terrain:
    """A level channel between two walls rising away from it."""
    half = max(size / 2.0, 1e-6)
    run = max(half - floor_width / 2.0, 1e-6)

    def height(x: float, z: float) -> float:
        across = abs(x) - floor_width / 2.0
        if across <= 0.0:
            return 0.0
        return depth * min(across / run, 1.0)

    return Terrain(size=size, tiles=tiles, height=height, material=material)


def flattened(
    terrain: Terrain, spots: Sequence[Spot], width: float, blend: float = 6.0
) -> Terrain:
    """Level the ground along a path, easing back into the surrounding shape.

    What makes a road or a racing line usable on uneven terrain: the surface
    under it is flat, and the hills resume `blend` metres to either side.
    """
    base = terrain.height
    if base is None or not spots:
        return terrain
    level = sum(base(x, z) for x, z in spots) / len(spots)

    def height(x: float, z: float) -> float:
        near = min(math.dist((x, z), spot) for spot in spots)
        if near <= width:
            return level
        if near >= width + blend:
            return base(x, z)
        eased = (near - width) / blend
        return level + (base(x, z) - level) * eased

    return replace(terrain, height=height)


def ground_height(terrain: Terrain, x: float, z: float) -> float:
    """Surface height at one point."""
    if terrain.height is None:
        return 0.0
    return float(terrain.height(x, z))


def tile_centres(terrain: Terrain) -> list[Spot]:
    """Centre of every tile in the terrain grid."""
    step = terrain.size / terrain.tiles
    origin = -terrain.size / 2.0 + step / 2.0
    return [
        (origin + column * step, origin + row * step)
        for row in range(terrain.tiles)
        for column in range(terrain.tiles)
    ]


def terrain_parts(terrain: Terrain) -> list[dict[str, Any]]:
    """Spec parts for the ground: one box when flat, a grid of tiles when not.

    Every tile is grown down to a shared floor rather than given a fixed
    thickness, so terrain dipping below zero — a basin, a canyon — stays a
    solid block instead of asking for a box of negative height.
    """
    if terrain.height is None or terrain.tiles <= 1:
        return [{
            "id": "ground",
            "kind": "box",
            "at": (0.0, -terrain.thickness / 2.0, 0.0),
            "size": (terrain.size, terrain.thickness, terrain.size),
            "material": terrain.material,
        }]

    step = terrain.size / terrain.tiles
    tops = {spot: ground_height(terrain, *spot) for spot in tile_centres(terrain)}
    floor = min(tops.values()) - terrain.thickness

    parts = []
    for row in range(terrain.tiles):
        for column in range(terrain.tiles):
            spot = (
                -terrain.size / 2.0 + step / 2.0 + column * step,
                -terrain.size / 2.0 + step / 2.0 + row * step,
            )
            top = tops[spot]
            depth = top - floor
            parts.append({
                "id": f"ground-{row:02d}-{column:02d}",
                "kind": "box",
                "at": (spot[0], floor + depth / 2.0, spot[1]),
                "size": (step, depth, step),
                "material": terrain.material,
            })
    return parts


# ── layout ───────────────────────────────────────────────────────────────────

def grid_spots(
    rows: int, columns: int, spacing: float, centre: Spot = (0.0, 0.0)
) -> list[Spot]:
    """Evenly spaced rows and columns, centred on `centre`."""
    x0 = centre[0] - (columns - 1) * spacing / 2.0
    z0 = centre[1] - (rows - 1) * spacing / 2.0
    return [
        (x0 + column * spacing, z0 + row * spacing)
        for row in range(rows)
        for column in range(columns)
    ]


def ring_spots(
    count: int, radius: float, centre: Spot = (0.0, 0.0), start_degrees: float = 0.0
) -> list[Spot]:
    """Points spread evenly around a circle."""
    spots = []
    for index in range(count):
        angle = math.radians(start_degrees + 360.0 * index / max(count, 1))
        spots.append(
            (centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle))
        )
    return spots


def line_spots(
    count: int,
    spacing: float,
    angle_degrees: float = 0.0,
    centre: Spot = (0.0, 0.0),
) -> list[Spot]:
    """Points along a straight line through `centre`."""
    angle = math.radians(angle_degrees)
    dx, dz = math.cos(angle), math.sin(angle)
    offset = (count - 1) * spacing / 2.0
    return [
        (
            centre[0] + (index * spacing - offset) * dx,
            centre[1] + (index * spacing - offset) * dz,
        )
        for index in range(count)
    ]


def scatter_spots(
    count: int,
    extent: float,
    seed: int = 0,
    min_gap: float = 2.0,
    centre: Spot = (0.0, 0.0),
    attempts: int = 40,
) -> list[Spot]:
    """Random points at least `min_gap` apart.

    Returns fewer than `count` when the area cannot hold that many at the
    requested spacing.
    """
    rng = random.Random(seed)
    half = extent / 2.0
    spots: list[Spot] = []
    for _ in range(count):
        for _ in range(attempts):
            spot = (
                centre[0] + rng.uniform(-half, half),
                centre[1] + rng.uniform(-half, half),
            )
            if all(math.dist(spot, other) >= min_gap for other in spots):
                spots.append(spot)
                break
    return spots


def clear_circle(
    spots: Iterable[Spot], centre: Spot = (0.0, 0.0), radius: float = 4.0
) -> list[Spot]:
    """Drop the spots inside a circle, leaving room for spawn or play space."""
    return [spot for spot in spots if math.dist(spot, centre) > radius]


def on_high_ground(
    spots: Iterable[Spot], terrain: Terrain, above: float
) -> list[Spot]:
    """Keep the spots where the ground stands at least `above` metres up.

    Ties a layout to the shape under it, so a settlement claims the ridges and
    a basin's rim rather than being scattered without regard to relief.
    """
    return [spot for spot in spots if ground_height(terrain, *spot) >= above]


def on_low_ground(
    spots: Iterable[Spot], terrain: Terrain, below: float
) -> list[Spot]:
    """Keep the spots where the ground sits at or under `below` metres."""
    return [spot for spot in spots if ground_height(terrain, *spot) <= below]


def on_slope(
    spots: Iterable[Spot], terrain: Terrain, steeper_than: float, step: float = 1.0
) -> list[Spot]:
    """Keep the spots where the ground falls faster than `steeper_than`.

    Measured as metres of fall per metre travelled, sampled `step` metres out
    in x and z.
    """
    kept = []
    for x, z in spots:
        here = ground_height(terrain, x, z)
        gradient = max(
            abs(ground_height(terrain, x + step, z) - here),
            abs(ground_height(terrain, x, z + step) - here),
        ) / step
        if gradient > steeper_than:
            kept.append((x, z))
    return kept


def clear_of(
    spots: Iterable[Spot],
    props: Sequence["Prop"],
    margin: float = 0.0,
) -> list[Spot]:
    """Drop the spots that land on something already placed.

    Compared against each prop's yawed footprint plus `margin`, so scenery can
    be scattered over a whole site and still miss the buildings.
    """
    kept = []
    for spot in spots:
        if all(not _inside(spot, prop, margin) for prop in props):
            kept.append(spot)
    return kept


def _inside(spot: Spot, prop: "Prop", margin: float) -> bool:
    half_x, half_z = footprint(prop)
    return (
        abs(spot[0] - prop.at[0]) <= half_x + margin
        and abs(spot[1] - prop.at[1]) <= half_z + margin
    )


# ── sizing ───────────────────────────────────────────────────────────────────

def uniform_sizes(count: int, size: Vec3) -> list[Vec3]:
    """The same size for every prop."""
    return [tuple(float(v) for v in size)] * count  # type: ignore[return-value]


def varied_sizes(
    count: int, base: Vec3, spread: float = 0.25, seed: int = 0
) -> list[Vec3]:
    """Sizes jittered around `base` by up to `spread` either way."""
    rng = random.Random(seed)
    sizes = []
    for _ in range(count):
        scale = 1.0 + rng.uniform(-spread, spread)
        sizes.append(tuple(float(v) * scale for v in base))
    return sizes  # type: ignore[return-value]


def graded_sizes(count: int, near: Vec3, far: Vec3) -> list[Vec3]:
    """Sizes stepping from `near` to `far` across the run."""
    if count == 1:
        return [tuple(float(v) for v in near)]  # type: ignore[return-value]
    sizes = []
    for index in range(count):
        t = index / (count - 1)
        sizes.append(tuple(float(a) + (float(b) - float(a)) * t
                           for a, b in zip(near, far)))
    return sizes  # type: ignore[return-value]


# ── props and scene ──────────────────────────────────────────────────────────

@dataclass
class Prop:
    """One object standing on the terrain.

    `at` is the ground-plane position; the height comes from the terrain, so
    moving a prop never leaves it floating. `source` replaces the primitive
    with a generated GLB and is what stage 2 fills in. Props sharing a `group`
    are expected to touch — road segments, wall runs, floor tiles — and are not
    reported against each other as collisions.
    """

    id: str
    kind: str = "box"
    at: Spot = (0.0, 0.0)
    size: Vec3 = (1.0, 1.0, 1.0)
    yaw: float = 0.0
    material: str = "block"
    source: str | None = None
    sink: float = 0.0
    group: str = ""


@dataclass
class Scene:
    """A terrain plus everything standing on it."""

    name: str
    terrain: Terrain = field(default_factory=Terrain)
    props: list[Prop] = field(default_factory=list)
    forward: str = "+z"
    materials: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(GREYBOX_MATERIALS)
    )


def place(
    prefix: str,
    kind: str,
    spots: Sequence[Spot],
    size: Vec3 | Sequence[Vec3] = (1.0, 1.0, 1.0),
    material: str = "block",
    yaw: float | Sequence[float] = 0.0,
    face_centre: bool = False,
) -> list[Prop]:
    """Turn positions into props, one per spot.

    `size` and `yaw` take either one value for all of them or one per spot.
    `face_centre` turns each prop to look at the origin, which is what a ring
    of seats or walls wants.
    """
    sizes = list(size) if _is_sequence_of_vec3(size) else [size] * len(spots)
    if face_centre:
        yaws = [math.degrees(math.atan2(x, z)) for x, z in spots]
    elif isinstance(yaw, (int, float)):
        yaws = [float(yaw)] * len(spots)
    else:
        yaws = list(yaw)

    return [
        Prop(
            id=f"{prefix}-{index:02d}",
            kind=kind,
            at=(float(spot[0]), float(spot[1])),
            size=tuple(float(v) for v in sizes[index]),  # type: ignore[arg-type]
            yaw=float(yaws[index]),
            material=material,
        )
        for index, spot in enumerate(spots)
    ]


def _is_sequence_of_vec3(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and isinstance(value[0], (list, tuple))
    )


def placed_height(prop: Prop) -> float:
    """How tall the prop ends up once written.

    A primitive is exactly its declared height. A mesh is fitted into `size`
    by a single factor to keep its proportions, so it reaches the declared
    height only on its longest axis and is shorter on the other two.
    """
    if not prop.source:
        return float(prop.size[1])

    from models.common.glb_writer import mesh_unit_extent

    return mesh_unit_extent(prop.source)[1] * max(float(v) for v in prop.size)


def prop_part(terrain: Terrain, prop: Prop) -> dict[str, Any]:
    """Spec part for one prop, resting on the terrain."""
    x, z = prop.at
    base = ground_height(terrain, x, z) - prop.sink
    part: dict[str, Any] = {
        "id": prop.id,
        "kind": "mesh" if prop.source else prop.kind,
        "at": (x, base + placed_height(prop) / 2.0, z),
        "size": tuple(float(v) for v in prop.size),
        "material": prop.material,
    }
    if prop.yaw:
        part["rotation"] = (0.0, float(prop.yaw), 0.0)
    if prop.source:
        part["source"] = prop.source
    return part


def build_spec(scene: Scene) -> dict[str, Any]:
    """Assemble the scene into a spec `write_spec_glb` can write."""
    return {
        "subject": scene.name,
        "units": UNITS,
        "forward": scene.forward,
        "materials": scene.materials,
        "parts": terrain_parts(scene.terrain)
        + [prop_part(scene.terrain, prop) for prop in scene.props],
    }


def write_scene(scene: Scene, out_path: str | Path) -> str:
    """Write the scene as a GLB and return the path."""
    from models.common.glb_writer import write_spec_glb

    return write_spec_glb(build_spec(scene), out_path)


# ── validation ───────────────────────────────────────────────────────────────

def footprint(prop: Prop) -> tuple[float, float]:
    """Half-extents in x and z of the upright box around the turned prop.

    Used to keep scattered spots off what is already placed. `overlap` measures
    the turned shape itself, so this only needs to be an outer bound.
    """
    half_x, half_z = prop.size[0] / 2.0, prop.size[2] / 2.0
    angle = math.radians(prop.yaw)
    cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
    return (half_x * cos + half_z * sin, half_x * sin + half_z * cos)


def bounds(terrain: Terrain, prop: Prop) -> tuple[Vec3, Vec3]:
    """World-space minimum and maximum corners of a placed prop.

    A prop carrying a `source` is measured by the writer's own
    `rotated_bounds`, because a mesh is fitted into `size` by a single factor
    and so fills the box on its longest axis only. Measuring it as the full
    box would validate an arrangement that is not the one written.
    """
    part = prop_part(terrain, prop)
    if prop.source:
        from models.common.glb_writer import rotated_bounds

        return rotated_bounds(
            part["size"], part["at"], part.get("rotation", (0.0, 0.0, 0.0)),
            kind="mesh", source=prop.source,
        )

    x, z = prop.at
    half_x, half_z = footprint(prop)
    base = ground_height(terrain, x, z) - prop.sink
    return (
        (x - half_x, base, z - half_z),
        (x + half_x, base + prop.size[1], z + half_z),
    )


def ground_corners(terrain: Terrain, prop: Prop) -> list[Spot]:
    """The prop's four ground-plane corners, turned by its yaw.

    Turned with the writer's own `euler_matrix`, so the shape measured here is
    the shape written. A prop carrying a `source` is squared off instead, since
    the writer reports a mesh's extent as an axis-aligned box.
    """
    if prop.source:
        low, high = bounds(terrain, prop)
        return [
            (low[0], low[2]), (high[0], low[2]),
            (high[0], high[2]), (low[0], high[2]),
        ]

    from models.common.glb_writer import apply_matrix, euler_matrix

    matrix = euler_matrix((0.0, float(prop.yaw), 0.0))
    half_x, half_z = prop.size[0] / 2.0, prop.size[2] / 2.0
    corners = []
    for dx, dz in ((-half_x, -half_z), (half_x, -half_z),
                   (half_x, half_z), (-half_x, half_z)):
        turned = apply_matrix(matrix, (dx, 0.0, dz))
        corners.append((prop.at[0] + turned[0], prop.at[1] + turned[2]))
    return corners


def _plane_overlap(first: list[Spot], second: list[Spot]) -> float:
    """Least distance the two quads would move to separate, or 0 when apart.

    Tested against every edge normal of both shapes, so a turned box is
    measured as itself rather than as the larger upright box around it — which
    would report clashes between neighbours that are comfortably clear.
    """
    least = float("inf")
    for corners in (first, second):
        for index in range(len(corners)):
            x1, z1 = corners[index]
            x2, z2 = corners[(index + 1) % len(corners)]
            length = math.hypot(x2 - x1, z2 - z1)
            if length < 1e-9:
                continue
            axis = (-(z2 - z1) / length, (x2 - x1) / length)

            reach_a = [x * axis[0] + z * axis[1] for x, z in first]
            reach_b = [x * axis[0] + z * axis[1] for x, z in second]
            gap = min(max(reach_a) - min(reach_b), max(reach_b) - min(reach_a))
            if gap <= 0.0:
                return 0.0
            least = min(least, gap)
    return 0.0 if least == float("inf") else least


def overlap(terrain: Terrain, first: Prop, second: Prop) -> tuple[float, float] | None:
    """How far two props interpenetrate on the ground and in height.

    Returns None when they are clear in either, so a roof resting on a house or
    a canopy above a trunk reads as separation rather than a collision.
    """
    low_a, high_a = bounds(terrain, first)
    low_b, high_b = bounds(terrain, second)
    vertical = min(high_a[1], high_b[1]) - max(low_a[1], low_b[1])
    if vertical <= 0.0:
        return None

    plane = _plane_overlap(
        ground_corners(terrain, first), ground_corners(terrain, second)
    )
    return (plane, vertical) if plane > 0.0 else None


def check_scene(scene: Scene, tolerance: float = 0.01) -> list[str]:
    """Report what would make the greybox unusable.

    Catches the faults that survive into a finished scene: props off the
    terrain, props inside one another, sizes that read wrong against a human,
    and duplicate ids that would collapse two nodes into one after export.
    `tolerance` is how much interpenetration counts as touching rather than
    clashing.
    """
    problems: list[str] = []
    half = scene.terrain.size / 2.0

    seen: set[str] = set()
    for prop in scene.props:
        if prop.id in seen:
            problems.append(f"{prop.id}: duplicate id")
        seen.add(prop.id)

        if any(value <= 0 for value in prop.size):
            problems.append(f"{prop.id}: size {prop.size} must be positive in every axis")
            continue

        low, high = bounds(scene.terrain, prop)
        if min(low[0], low[2]) < -half or max(high[0], high[2]) > half:
            problems.append(
                f"{prop.id}: spans x {low[0]:.1f}..{high[0]:.1f}, "
                f"z {low[2]:.1f}..{high[2]:.1f}, past the "
                f"{scene.terrain.size:.0f} m terrain"
            )

        height = prop.size[1]
        if height > HUMAN_HEIGHT * 20:
            problems.append(
                f"{prop.id}: {height:.1f} m tall, over 20x a person — check the units"
            )
        elif height < HUMAN_HEIGHT / 20:
            problems.append(
                f"{prop.id}: {height:.2f} m tall, under a twentieth of a person"
            )

    for index, prop in enumerate(scene.props):
        for other in scene.props[index + 1:]:
            if prop.group and prop.group == other.group:
                continue
            depths = overlap(scene.terrain, prop, other)
            if depths and min(depths) > tolerance:
                problems.append(
                    f"{prop.id} and {other.id}: overlap by {depths[0]:.2f} m "
                    f"on the ground, {depths[1]:.2f} m in height"
                )

    return problems


def scene_summary(scene: Scene) -> dict[str, Any]:
    """Counts and extents, for a report or a test assertion."""
    heights = [prop.size[1] for prop in scene.props] or [0.0]
    return {
        "name": scene.name,
        "terrain_size": scene.terrain.size,
        "terrain_tiles": scene.terrain.tiles,
        "props": len(scene.props),
        "kinds": sorted({prop.kind for prop in scene.props}),
        "materials": sorted({prop.material for prop in scene.props}),
        "tallest": max(heights),
        "generated_parts": sum(1 for prop in scene.props if prop.source),
    }


# ── stage 2: detail ──────────────────────────────────────────────────────────

def swap_mesh(
    scene: Scene, prop_id: str, source: str | Path, height: float | None = None
) -> Scene:
    """Replace one greybox prop with a generated mesh, keeping its placement.

    The position is what the greybox already validated, so the mesh drops into
    the spot the block occupied.

    `height` is how tall the mesh should end up. Without it the prop's declared
    box is used, and because a mesh is fitted by a single factor that means its
    *largest* dimension — swapping a figure into a wide flat dais would scale
    the figure to the dais's width. State the height whenever the mesh is not
    the same shape as the block it replaces.
    """
    if not any(prop.id == prop_id for prop in scene.props):
        raise KeyError(f"no prop {prop_id!r} in scene {scene.name!r}")

    size = None
    if height is not None:
        from models.common.glb_writer import mesh_unit_extent

        extent = mesh_unit_extent(str(source))[1]
        if extent <= 0:
            raise ValueError(f"{source} has no height to scale")
        scale = float(height) / extent
        size = (scale, scale, scale)

    return replace(
        scene,
        props=[
            replace(prop, source=str(source), **({"size": size} if size else {}))
            if prop.id == prop_id
            else prop
            for prop in scene.props
        ],
    )


def swap_material(scene: Scene, material: str, colour: Sequence[float],
                  roughness: float = 0.8, metallic: float = 0.0) -> Scene:
    """Recolour one material across the whole scene."""
    colours = list(colour)
    if len(colours) == 3:
        colours.append(1.0)
    materials = dict(scene.materials)
    materials[material] = {
        "baseColor": [float(v) for v in colours],
        "roughness": float(roughness),
        "metallic": float(metallic),
    }
    return replace(scene, materials=materials)


def texture_ids(scene: Scene) -> list[str]:
    """Catalogue asset ids covering the materials this scene uses."""
    used = {scene.terrain.material} | {prop.material for prop in scene.props}
    return sorted({MATERIAL_TEXTURES[name] for name in used if name in MATERIAL_TEXTURES})


def stage_scene_textures(
    project_dir: str | Path, scene: Scene, cache_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Copy this scene's ground and wall textures into a game project.

    Returns one record per staged file, as `scene_assets.stage` does.
    """
    from operators.gen_3d_scene.funcs import scene_assets

    return [
        scene_assets.stage(Path(project_dir), asset_id,
                           cache_dir=Path(cache_dir) if cache_dir else None)
        for asset_id in texture_ids(scene)
    ]


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import argparse

    from operators.gen_3d_scene.funcs.terrain_code_template import TEMPLATES

    parser = argparse.ArgumentParser(description="Write greybox scenes as GLB.")
    parser.add_argument("--templates", nargs="*", default=sorted(TEMPLATES))
    parser.add_argument(
        "--out",
        default="test_data/outputs/greybox_scenes",
        help="directory to write into, relative to the repository root",
    )
    options = parser.parse_args()

    directory = Path(options.out)
    directory.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in options.templates:
        scene = TEMPLATES[name]()
        problems = check_scene(scene)
        path = write_scene(scene, directory / f"{name}.glb")
        summary = scene_summary(scene)
        print(
            f"{name:<12} {summary['props']:>3} props  "
            f"{Path(path).stat().st_size / 1024:>7.0f} KB  "
            f"{len(problems)} problem(s)"
        )
        for line in problems:
            print(f"    {line}")
        failures += len(problems)
    raise SystemExit(1 if failures else 0)
