import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gpkg_tiles import GpkgTiles


class CoverageDtypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.output = Path(self.work.name) / "coverage.gpkg"
        sample = Path(__file__).parent / "example_geopackages" / "small_world_jpg_png.gpkg"
        self.source = GpkgTiles.open(sample, "small_world_jpg_png")
        self.shape = (self.source.tile_height, self.source.tile_width)

    def tearDown(self) -> None:
        self.source.close()
        self.work.cleanup()

    def test_identity_transform_returns_uint16(self) -> None:
        with GpkgTiles.create(
            self.output, "unsigned", like=self.source, kind="coverage",
            dtype=np.uint16, data_null=65535, scale=1, offset=0, precision=1,
        ) as coverage:
            expected = np.ma.array(
                np.full(self.shape, 1234, dtype=np.uint16), mask=False
            )
            expected.mask[:2, :2] = True
            coverage.put(coverage.base_zoom, 0, 0, expected)
            actual = coverage.get(coverage.base_zoom, 0, 0)
            self.assertEqual(coverage.dtype, np.dtype(np.uint16))
            self.assertEqual(actual.dtype, np.dtype(np.uint16))
            np.testing.assert_array_equal(actual, expected)

    def test_signed_transform_returns_int16(self) -> None:
        with GpkgTiles.create(
            self.output, "signed", like=self.source, kind="coverage",
            dtype=np.uint16, data_null=65535, scale=1, offset=-32768, precision=1,
        ) as coverage:
            expected = np.full(self.shape, -1234, dtype=np.int16)
            coverage.put(coverage.base_zoom, 0, 0, expected)
            actual = coverage.get(coverage.base_zoom, 0, 0)
            self.assertEqual(coverage.dtype, np.dtype(np.int16))
            self.assertEqual(actual.dtype, np.dtype(np.int16))
            np.testing.assert_array_equal(actual, expected)

    def test_scale_returns_float32(self) -> None:
        with GpkgTiles.create(
            self.output, "scaled", like=self.source, kind="coverage",
            dtype=np.uint16, data_null=65535, scale=0.1, offset=0, precision=0.1,
        ) as coverage:
            expected = np.full(self.shape, 12.3, dtype=np.float32)
            coverage.put(coverage.base_zoom, 0, 0, expected)
            coverage.commit()
            actual = coverage.get(coverage.base_zoom, 0, 0)
            self.assertEqual(coverage.dtype, np.dtype(np.float32))
            self.assertEqual(actual.dtype, np.dtype(np.float32))
            np.testing.assert_allclose(actual, expected, atol=1e-6)

            # Cloning preserves the integer storage encoding even though the
            # decoded values are Float32.
            with GpkgTiles.create(self.output, "scaled_clone", like=coverage) as clone:
                self.assertEqual(clone.dtype, np.dtype(np.float32))

    def test_non_identity_tile_transform_promotes_layer(self) -> None:
        with GpkgTiles.create(
            self.output, "tile_scaled", like=self.source, kind="coverage",
            dtype=np.uint16, data_null=65535, scale=1, offset=0, precision=0.01,
        ) as coverage:
            coverage.put(
                coverage.base_zoom, 0, 0,
                np.full(self.shape, 1, dtype=np.uint16),
            )
            coverage.put(
                coverage.base_zoom, 1, 0,
                np.full(self.shape, 1.25, dtype=np.float32),
            )
            actual = coverage.get(coverage.base_zoom, 0, 0)
            self.assertEqual(coverage.dtype, np.dtype(np.float32))
            self.assertEqual(actual.dtype, np.dtype(np.float32))


class ImageOverviewFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.output = Path(self.work.name) / "imagery.gpkg"
        sample = Path(__file__).parent / "example_geopackages" / "small_world_jpg_png.gpkg"
        self.source = GpkgTiles.open(sample, "small_world_jpg_png")

    def tearDown(self) -> None:
        self.source.close()
        self.work.cleanup()

    def test_overview_image_format_policy(self) -> None:
        with GpkgTiles.create(self.output, "imagery", like=self.source) as imagery:
            # Make this test grid fully covered vertically. The public sample's
            # coarse matrix intentionally extends beyond its single source row.
            imagery._matrix[0]["pixel_y_size"] = imagery._matrix[1]["pixel_y_size"]
            size = (imagery.tile_width, imagery.tile_height)
            imagery.put(imagery.base_zoom, 0, 0, Image.new("RGB", size, "red"))
            imagery.put(imagery.base_zoom, 1, 0, Image.new("RGBA", size, (0, 0, 255, 0)))

            imagery.build_overviews()
            webp = imagery.get(0, 0, 0)
            self.assertEqual(webp.format, "WEBP")
            self.assertEqual(webp.getchannel("A").getextrema()[0], 0)
            extension = imagery._connection().execute(
                "SELECT 1 FROM gpkg_extensions WHERE table_name=? "
                "AND column_name='tile_data' AND extension_name='gpkg_webp'",
                (imagery.table,),
            ).fetchone()
            self.assertIsNotNone(extension)

            imagery.build_overviews(image_format="jpeg")
            transparent = imagery.get(0, 0, 0)
            self.assertEqual(transparent.format, "PNG")
            self.assertEqual(transparent.mode, "RGBA")

            imagery.put(imagery.base_zoom, 1, 0, Image.new("RGB", size, "blue"))
            imagery.build_overviews(image_format="jpeg")
            opaque = imagery.get(0, 0, 0)
            self.assertEqual(opaque.format, "JPEG")
            self.assertEqual(opaque.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
