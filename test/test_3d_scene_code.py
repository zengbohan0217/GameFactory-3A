"""
test/test_3d_scene_code.py

Tests for the code-built `3d_scene` route: terrain, layout and sizing written
as Python, checked as a greybox, then detailed.

Nothing here needs weights, a GPU or a network, so it runs anywhere. The scene
geometry is compared against `models/common/glb_writer`, which is what actually
writes the file — a validator that disagrees with the writer passes scenes that
are wrong on disk.

Run from repo root:
    python test/test_3d_scene_code.py
    python test/test_3d_scene_code.py --video      # also record turntables
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models.common.glb_writer import rotated_bounds  # noqa: E402
from operators.gen_3d_scene.funcs import terrain_code_edit as te  # noqa: E402
from operators.gen_3d_scene.funcs.terrain_code_template import TEMPLATES  # noqa: E402

OUT_DIR = _REPO_ROOT / "test_data" / "outputs" / "_test_3d_scene_code"
VIEWER_LIB = _REPO_ROOT / "test_data" / "outputs" / "_viewer_lib"


def glb_json(path: Path) -> dict:
    """The JSON chunk of a GLB, for checking what was actually written."""
    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if kind == 0x4E4F534A:
            return json.loads(data[offset + 8: offset + 8 + length])
        offset += 8 + length
    raise AssertionError(f"{path} has no JSON chunk")


# ── terrain ──────────────────────────────────────────────────────────────────

class TestTerrainShapes(unittest.TestCase):
    """Each terrain function has to produce the landform it is named for."""

    def test_terrain_flat_is_level_everywhere(self):
        ground = te.flat(40.0)
        for x, z in ((0.0, 0.0), (15.0, -12.0), (-19.0, 19.0)):
            self.assertEqual(te.ground_height(ground, x, z), 0.0)

    def test_terrain_bowl_is_lowest_at_the_centre(self):
        ground = te.bowl(60.0, depth=8.0)
        centre = te.ground_height(ground, 0.0, 0.0)
        self.assertAlmostEqual(centre, -8.0, places=6)
        for radius in (10.0, 20.0, 29.0):
            self.assertGreater(te.ground_height(ground, radius, 0.0), centre)
        self.assertAlmostEqual(te.ground_height(ground, 30.0, 0.0), 0.0, places=6)

    def test_terrain_mound_has_a_level_top(self):
        ground = te.mound(60.0, rise=6.0, flat_radius=12.0)
        for radius in (0.0, 6.0, 11.9):
            self.assertAlmostEqual(te.ground_height(ground, radius, 0.0), 6.0,
                                   places=6)
        self.assertLess(te.ground_height(ground, 20.0, 0.0), 6.0)
        self.assertAlmostEqual(te.ground_height(ground, 30.0, 0.0), 0.0, places=6)

    def test_terrain_canyon_has_a_level_floor_between_walls(self):
        ground = te.canyon(70.0, depth=9.0, floor_width=16.0)
        for x in (-7.0, 0.0, 7.0):
            self.assertEqual(te.ground_height(ground, x, 5.0), 0.0)
        self.assertGreater(te.ground_height(ground, 20.0, 5.0), 0.0)
        self.assertAlmostEqual(te.ground_height(ground, 35.0, 5.0), 9.0, places=6)

    def test_terrain_flattened_levels_a_path_and_eases_back(self):
        rolling = te.hills(60.0, amplitude=4.0, wavelength=20.0)
        path = [(0.0, z) for z in (-10.0, 0.0, 10.0)]
        levelled = te.flattened(rolling, path, width=4.0, blend=6.0)

        heights = {te.ground_height(levelled, 0.0, z) for z in (-10.0, 0.0, 10.0)}
        self.assertEqual(len(heights), 1, "the path itself must be level")

        far = 30.0
        self.assertAlmostEqual(
            te.ground_height(levelled, far, 0.0),
            te.ground_height(rolling, far, 0.0),
            places=6,
        )

    def test_terrain_tiles_never_have_negative_depth(self):
        """Terrain dipping below zero still has to produce solid boxes."""
        for name, ground in (
            ("bowl", te.bowl(60.0, depth=9.0)),
            ("canyon", te.canyon(70.0, depth=9.0)),
            ("hills", te.hills(60.0, amplitude=5.0)),
            ("mound", te.mound(60.0, rise=6.0)),
        ):
            with self.subTest(terrain=name):
                for part in te.terrain_parts(ground):
                    self.assertTrue(
                        all(value > 0 for value in part["size"]),
                        f"{name} tile {part['id']} has size {part['size']}",
                    )


# ── layout ───────────────────────────────────────────────────────────────────

class TestLayoutHelpers(unittest.TestCase):
    """Spot generators and the filters that tie a layout to the ground."""

    def test_layout_ring_spots_sit_on_the_circle(self):
        for spot in te.ring_spots(8, 12.0):
            self.assertAlmostEqual(math.hypot(*spot), 12.0, places=6)

    def test_layout_grid_spots_are_centred_and_evenly_spaced(self):
        spots = te.grid_spots(3, 3, 5.0)
        self.assertEqual(len(spots), 9)
        self.assertIn((0.0, 0.0), spots)
        xs = sorted({round(x, 6) for x, _ in spots})
        self.assertEqual(xs, [-5.0, 0.0, 5.0])

    def test_layout_scatter_spots_respect_the_minimum_gap(self):
        spots = te.scatter_spots(12, 40.0, seed=5, min_gap=6.0)
        for index, first in enumerate(spots):
            for second in spots[index + 1:]:
                self.assertGreaterEqual(math.dist(first, second), 6.0)

    def test_layout_scatter_spots_are_reproducible(self):
        self.assertEqual(
            te.scatter_spots(10, 30.0, seed=7),
            te.scatter_spots(10, 30.0, seed=7),
        )

    def test_layout_height_filters_split_by_elevation(self):
        ground = te.hills(60.0, amplitude=5.0, wavelength=25.0)
        spots = te.scatter_spots(60, 50.0, seed=3, min_gap=3.0)

        high = te.on_high_ground(spots, ground, above=2.0)
        low = te.on_low_ground(spots, ground, below=-2.0)

        self.assertTrue(high and low, "the filters must each keep something")
        self.assertFalse(set(high) & set(low))
        for spot in high:
            self.assertGreaterEqual(te.ground_height(ground, *spot), 2.0)
        for spot in low:
            self.assertLessEqual(te.ground_height(ground, *spot), -2.0)

    def test_layout_slope_filter_keeps_only_steep_ground(self):
        ground = te.canyon(70.0, depth=9.0, floor_width=16.0)
        floor = [(0.0, 0.0), (4.0, 10.0)]
        wall = [(24.0, 0.0), (-26.0, 8.0)]

        self.assertEqual(te.on_slope(floor, ground, steeper_than=0.05), [])
        self.assertEqual(len(te.on_slope(wall, ground, steeper_than=0.05)), 2)

    def test_layout_clear_of_avoids_placed_props(self):
        blocker = te.Prop("blocker", "box", (0.0, 0.0), (8.0, 3.0, 8.0))
        spots = [(0.0, 0.0), (3.0, 3.0), (20.0, 20.0)]
        self.assertEqual(te.clear_of(spots, [blocker]), [(20.0, 20.0)])


# ── geometry ─────────────────────────────────────────────────────────────────

class TestSceneGeometry(unittest.TestCase):
    """What the checker measures must be what the writer writes."""

    def test_geom_corners_match_the_writer_at_every_yaw(self):
        ground = te.flat(60.0)
        for yaw in (0, 30, 45, 60, 90, 144, 210, -60):
            with self.subTest(yaw=yaw):
                prop = te.Prop("p", "box", (5.0, 3.0), (4.2, 2.0, 3.6),
                               yaw=float(yaw))
                corners = te.ground_corners(ground, prop)
                mine = (
                    min(c[0] for c in corners), max(c[0] for c in corners),
                    min(c[1] for c in corners), max(c[1] for c in corners),
                )
                part = te.prop_part(ground, prop)
                low, high = rotated_bounds(
                    part["size"], part["at"],
                    part.get("rotation", (0.0, 0.0, 0.0)), kind="box",
                )
                for got, want in zip(mine, (low[0], high[0], low[2], high[2])):
                    self.assertAlmostEqual(got, want, places=9)

    def test_geom_props_rest_on_the_terrain(self):
        ground = te.hills(60.0, amplitude=4.0)
        for spot in ((0.0, 0.0), (11.0, -7.0), (-14.0, 13.0)):
            prop = te.Prop("p", "box", spot, (2.0, 3.0, 2.0))
            low, _ = te.bounds(ground, prop)
            self.assertAlmostEqual(low[1], te.ground_height(ground, *spot),
                                   places=6)

    def test_geom_overlap_measures_the_turned_shape(self):
        ground = te.flat(50.0)

        def box(name, at, size, yaw=0.0):
            return te.Prop(name, "box", at, size, yaw=yaw)

        clear = te.overlap(ground, box("a", (0, 0), (2, 2, 2)),
                           box("b", (5, 0), (2, 2, 2)))
        self.assertIsNone(clear)

        through = te.overlap(ground, box("a", (0, 0), (2, 2, 2)),
                             box("b", (1, 0), (2, 2, 2)))
        self.assertIsNotNone(through)
        self.assertAlmostEqual(through[0], 1.0, places=6)

        touching = te.overlap(ground, box("a", (0, 0), (2, 2, 2)),
                              box("b", (2, 0), (2, 2, 2)))
        self.assertIsNone(touching)

        # An upright box around either shape would report a clash here.
        turned = te.overlap(ground, box("a", (0, 0), (4, 3, 3), 45.0),
                            box("b", (4.6, 0), (4, 3, 3), 45.0))
        self.assertIsNone(turned)

    def test_geom_stacked_props_are_separation_not_collision(self):
        ground = te.flat(40.0)
        base = te.Prop("base", "box", (0.0, 0.0), (4.0, 3.0, 4.0))
        on_top = te.Prop("top", "cone", (0.0, 0.0), (4.0, 2.0, 4.0), sink=-3.0)
        self.assertIsNone(te.overlap(ground, base, on_top))

    def test_geom_grouped_props_may_touch(self):
        ground = te.flat(40.0)
        scene = te.Scene("run", ground, [
            te.Prop("a", "box", (0.0, 0.0), (4.0, 1.0, 2.0), group="road"),
            te.Prop("b", "box", (3.5, 0.0), (4.0, 1.0, 2.0), group="road"),
        ])
        self.assertEqual(te.check_scene(scene), [])

        scene.props[1].group = "other"
        self.assertEqual(len(te.check_scene(scene)), 1)


# ── validation ───────────────────────────────────────────────────────────────

class TestSceneChecks(unittest.TestCase):
    """`check_scene` has to catch what makes a greybox unusable."""

    def test_check_reports_a_prop_off_the_terrain(self):
        scene = te.Scene("t", te.flat(20.0),
                         [te.Prop("far", "box", (30.0, 0.0), (2.0, 2.0, 2.0))])
        self.assertIn("terrain", " ".join(te.check_scene(scene)))

    def test_check_reports_duplicate_ids(self):
        scene = te.Scene("t", te.flat(40.0), [
            te.Prop("same", "box", (-8.0, 0.0), (2.0, 2.0, 2.0)),
            te.Prop("same", "box", (8.0, 0.0), (2.0, 2.0, 2.0)),
        ])
        self.assertIn("duplicate id", " ".join(te.check_scene(scene)))

    def test_check_reports_sizes_that_misread_against_a_person(self):
        giant = te.Scene("t", te.flat(400.0),
                         [te.Prop("g", "box", (0.0, 0.0), (2.0, 90.0, 2.0))])
        self.assertIn("check the units", " ".join(te.check_scene(giant)))

        speck = te.Scene("t", te.flat(40.0),
                         [te.Prop("s", "box", (0.0, 0.0), (0.05, 0.05, 0.05))])
        self.assertIn("twentieth", " ".join(te.check_scene(speck)))

    def test_check_reports_a_non_positive_size(self):
        scene = te.Scene("t", te.flat(40.0),
                         [te.Prop("flat", "box", (0.0, 0.0), (2.0, 0.0, 2.0))])
        self.assertIn("positive", " ".join(te.check_scene(scene)))


# ── templates ────────────────────────────────────────────────────────────────

class TestTemplates(unittest.TestCase):
    """Every landform template, and the differences between them."""

    def test_template_every_scene_is_clean(self):
        for name, build in TEMPLATES.items():
            with self.subTest(template=name):
                problems = te.check_scene(build())
                self.assertEqual(problems, [], f"{name}: {problems}")

    def test_template_names_match_the_scene_they_build(self):
        for name, build in TEMPLATES.items():
            with self.subTest(template=name):
                self.assertEqual(build().name, name)

    def test_template_is_reproducible(self):
        for name, build in TEMPLATES.items():
            with self.subTest(template=name):
                first = te.scene_summary(build())
                second = te.scene_summary(build())
                self.assertEqual(first, second)

    def test_template_terrain_profiles_differ(self):
        """The classification is by landform, so the ground must actually differ."""
        profiles = {}
        for name, build in TEMPLATES.items():
            ground = build().terrain
            samples = [
                round(te.ground_height(ground, x, 0.0), 3)
                for x in (-20.0, -10.0, 0.0, 10.0, 20.0)
            ]
            profiles[name] = samples

        # plains and city are both level, and are distinguished by layout
        # instead; every other pair has a different cross-section.
        shapes = {name: tuple(values) for name, values in profiles.items()}
        self.assertEqual(shapes["plains"], shapes["city"])
        relief = {n: s for n, s in shapes.items() if n not in ("plains", "city")}
        self.assertEqual(len(set(relief.values())), len(relief), relief)

    def test_template_flat_landforms_differ_by_layout(self):
        """`plains` and `city` share a terrain, so their spread must diverge."""
        scattered = TEMPLATES["plains"]()
        gridded = TEMPLATES["city"]()

        def spacings(scene):
            spots = [prop.at for prop in scene.props]
            return [
                min(math.dist(spot, other) for other in spots if other != spot)
                for spot in spots
            ]

        grid_gaps = spacings(gridded)
        self.assertLessEqual(max(grid_gaps) - min(grid_gaps), 0.5,
                             "a grid should have one spacing")
        scatter_gaps = spacings(scattered)
        self.assertGreater(max(scatter_gaps) - min(scatter_gaps), 1.0,
                           "a scatter should not")

    def test_template_hills_layout_follows_elevation(self):
        """Towers take the ridges and dwellings the hollows, or the split is fake."""
        scene = TEMPLATES["hills"]()
        ground = scene.terrain

        towers = [p for p in scene.props if p.id.startswith("tower")]
        huts = [p for p in scene.props if p.id.startswith("hut")]
        self.assertTrue(towers and huts)

        highest_hut = max(te.ground_height(ground, *p.at) for p in huts)
        lowest_tower = min(te.ground_height(ground, *p.at) for p in towers)
        self.assertGreater(lowest_tower, highest_hut)

    def test_template_canyon_debris_sits_on_the_walls(self):
        scene = TEMPLATES["canyon"]()
        rocks = [p for p in scene.props if p.id.startswith("rock")]
        self.assertTrue(rocks)
        for rock in rocks:
            self.assertGreater(te.ground_height(scene.terrain, *rock.at), 0.0)

    def test_template_basin_rings_face_the_centre(self):
        scene = TEMPLATES["basin"]()
        houses = [p for p in scene.props if p.id.startswith("house-")]
        self.assertTrue(houses)
        for house in houses:
            expected = math.degrees(math.atan2(house.at[0], house.at[1]))
            self.assertAlmostEqual(house.yaw, expected, places=6)


# ── writing ──────────────────────────────────────────────────────────────────

class TestSceneWriting(unittest.TestCase):
    """The GLB on disk has to match the scene that was checked."""

    def test_write_produces_a_readable_glb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(te.write_scene(TEMPLATES["plains"](),
                                       Path(directory) / "scene.glb"))
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:4], b"glTF")
            document = glb_json(path)
            self.assertTrue(document.get("meshes"))
            self.assertTrue(document.get("nodes"))

    def test_write_includes_terrain_and_every_prop(self):
        scene = TEMPLATES["city"]()
        expected = len(te.terrain_parts(scene.terrain)) + len(scene.props)
        self.assertEqual(len(te.build_spec(scene)["parts"]), expected)

    def test_write_every_template(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, build in TEMPLATES.items():
                with self.subTest(template=name):
                    path = Path(te.write_scene(build(),
                                               Path(directory) / f"{name}.glb"))
                    self.assertGreater(path.stat().st_size, 1000)


# ── detail ───────────────────────────────────────────────────────────────────

class TestSceneDetail(unittest.TestCase):
    """Stage 2: meshes and materials replacing greybox stand-ins."""

    MESH = (_REPO_ROOT / "test_data" / "outputs" / "game_knight_demo" / "default"
            / "assets" / "3d_object" / "knight_hybrid_001" / "model.glb")

    def setUp(self):
        if not self.MESH.is_file():
            self.skipTest(f"no generated mesh at {self.MESH}")

    def test_detail_swap_keeps_the_position(self):
        scene = TEMPLATES["plains"]()
        target = scene.props[0]
        swapped = te.swap_mesh(scene, target.id, self.MESH, height=2.0)
        replaced = next(p for p in swapped.props if p.id == target.id)
        self.assertEqual(replaced.at, target.at)
        self.assertEqual(replaced.source, str(self.MESH))

    def test_detail_swap_honours_the_requested_height(self):
        scene = TEMPLATES["plains"]()
        target = scene.props[0].id
        for height in (1.2, 2.5, 4.0):
            with self.subTest(height=height):
                swapped = te.swap_mesh(scene, target, self.MESH, height=height)
                prop = next(p for p in swapped.props if p.id == target)
                self.assertAlmostEqual(te.placed_height(prop), height, places=4)

    def test_detail_swapped_mesh_rests_on_the_ground(self):
        scene = TEMPLATES["hills"]()
        target = scene.props[0]
        swapped = te.swap_mesh(scene, target.id, self.MESH, height=3.0)
        prop = next(p for p in swapped.props if p.id == target.id)
        low, high = te.bounds(swapped.terrain, prop)
        self.assertAlmostEqual(low[1], te.ground_height(swapped.terrain, *prop.at),
                               places=4)
        self.assertAlmostEqual(high[1] - low[1], 3.0, places=4)

    def test_detail_swap_rejects_an_unknown_prop(self):
        with self.assertRaises(KeyError):
            te.swap_mesh(TEMPLATES["plains"](), "not-a-prop", self.MESH)

    def test_detail_material_swap_applies_everywhere(self):
        scene = te.swap_material(TEMPLATES["city"](), "wall", [0.2, 0.3, 0.4])
        self.assertEqual(scene.materials["wall"]["baseColor"],
                         [0.2, 0.3, 0.4, 1.0])

    def test_detail_texture_ids_cover_the_materials_used(self):
        for name, build in TEMPLATES.items():
            with self.subTest(template=name):
                scene = build()
                used = {scene.terrain.material} | {p.material for p in scene.props}
                expected = {te.MATERIAL_TEXTURES[m] for m in used
                            if m in te.MATERIAL_TEXTURES}
                self.assertEqual(set(te.texture_ids(scene)), expected)

    def test_detail_summary_counts_generated_parts(self):
        scene = TEMPLATES["plains"]()
        self.assertEqual(te.scene_summary(scene)["generated_parts"], 0)
        swapped = te.swap_mesh(scene, scene.props[0].id, self.MESH, height=2.0)
        self.assertEqual(te.scene_summary(swapped)["generated_parts"], 1)


# ── video ────────────────────────────────────────────────────────────────────

def record_videos(out_dir: Path = OUT_DIR, frames: int = 150) -> int:
    """Write every template as a GLB and record a turntable of each.

    Needs the compiled `turntable` helper in `test_data/outputs/_viewer_lib`
    and a local HTTP server rooted at `test_data/outputs`.
    """
    turntable = VIEWER_LIB / "turntable"
    if not turntable.is_file():
        print(f"no turntable helper at {turntable.relative_to(_REPO_ROOT)}")
        print("build it with:  cd test_data/outputs/_viewer_lib && "
              "swiftc -O -o turntable turntable.swift")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    videos = out_dir / "videos"
    videos.mkdir(exist_ok=True)

    outputs_root = _REPO_ROOT / "test_data" / "outputs"
    relative = out_dir.relative_to(outputs_root)

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8765", "--bind", "127.0.0.1"],
        cwd=outputs_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        import time

        time.sleep(2)
        for name, build in TEMPLATES.items():
            scene = build()
            problems = te.check_scene(scene)
            te.write_scene(scene, out_dir / f"{name}.glb")
            summary = te.scene_summary(scene)

            page = (
                "http://127.0.0.1:8765/_viewer_lib/scene_recorder.html"
                f"?model=../{relative}/{name}.glb"
                f"&label={name}"
                f"&note={summary['props']}+props+%C2%B7+"
                f"{len(problems)}+problem(s)"
                f"&frames={frames}"
            )
            subprocess.run(
                [str(turntable), "--url", page,
                 "--out", str(videos / f"{name}.mp4"),
                 "--frames", str(frames), "--fps", "30"],
                check=False,
            )
            written = videos / f"{name}.mp4"
            size = written.stat().st_size if written.is_file() else 0
            print(f"{name:<12} {summary['props']:>3} props  "
                  f"{len(problems)} problem(s)  {size / 1024:>7.0f} KB")
    finally:
        server.terminate()
        server.wait(timeout=5)

    print(f"\nvideos: {videos.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    if "--video" in sys.argv:
        sys.argv.remove("--video")
        raise SystemExit(record_videos())
    unittest.main(verbosity=2)
