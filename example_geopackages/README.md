# Example GeoPackages

Small public fixtures used by `gpkg_tiles_walkthrough.ipynb`.

| File | Source | SHA-256 |
| --- | --- | --- |
| `small_world_jpg_png.gpkg` | [GDAL sample data](https://download.osgeo.org/gdal/data/geopackage/small_world_jpg_png.gpkg) | `ea6f63ee85b335024f6e12408c2a7c535a56cf7b2ef9a62a305f4fecaefec8b4` |
| `small_world_png_with_ovr.gpkg` | [GDAL sample data](https://download.osgeo.org/gdal/data/geopackage/small_world_png_with_ovr.gpkg) | `c11a646b834b907ece7643490b197d63d7ff2b407febd4d3e3faf0475a67cfa9` |
| `small_world_webp.gpkg` | [GDAL sample data](https://download.osgeo.org/gdal/data/geopackage/small_world_webp.gpkg) | `525227d6b907b64d73c0919069151d7107490172ae46ec370cf68e447526e0be` |
| `stefan_full_rgba_missing_tiles.gpkg` | [GDAL sample data](https://download.osgeo.org/gdal/data/geopackage/stefan_full_rgba_missing_tiles.gpkg) | `ff2451fe45a01043267fb34a8ba657d5c9715952ceac268bc16543bcb141eeb3` |
| `rivers.gpkg` | [NGA GeoPackage examples](https://ngageoint.github.io/GeoPackage/examples/rivers.gpkg) | `d1df0b3bbd880168bf8d65bd686b781e4122cd14883ec6b3a4aa836c49c9cffe` |

The GDAL project states that its sample and autotest data are intended to be
public and freely redistributable. `rivers.gpkg` uses Natural Earth river data;
Natural Earth data is public domain. These files are retained unmodified.

The four older GDAL fixtures identify GeoPackage 1.0 with the legacy `GP10`
SQLite application ID. `rivers.gpkg` uses the GeoPackage 1.2+ `GPKG` ID.
