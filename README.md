# gpkg-tiles

`gpkg-tiles` is a small Python API for reading, creating, and transforming
image and gridded-coverage tile layers stored in a GeoPackage. It works at the
physical tile level: it does not reproject data or change the tile matrix set.

## Install from GitHub

Install the current default branch by replacing `OWNER` with the GitHub account
that hosts this repository:

```shell
python -m pip install "gpkg-tiles @ git+https://github.com/OWNER/gpkg-tiles.git"
```

For reproducible installations, pin a release tag or commit:

```shell
python -m pip install "gpkg-tiles @ git+https://github.com/OWNER/gpkg-tiles.git@v0.1.0"
```

The project requires Python 3.10 or newer. NumPy and Pillow are installed as
dependencies.

## Usage

Open a tile layer with `GpkgTiles.open()`. The returned object is either a
`GpkgImageTiles` or `GpkgGriddedCoverage`, based on the layer metadata.

```python
from gpkg_tiles import GpkgTiles

with GpkgTiles.open("source.gpkg", "imagery") as tiles:
    print(tiles.kind, tiles.zooms, tiles.bounds)
    for z, x, y, image in tiles.tiles():
        print(z, x, y, image.size)
```

Create a layer on the same grid as an existing layer and write transformed
tiles to it:

```python
from gpkg_tiles import GpkgTiles

with (
    GpkgTiles.open("source.gpkg", "imagery") as source,
    GpkgTiles.create(
        "result.gpkg",
        "grayscale",
        like=source,
        commit_every=500,
    ) as destination,
):
    for z, x, y, image in source.tiles("all"):
        destination.put(z, x, y, image.convert("L"))
```

Coverage creation supports `numpy.uint16` (16-bit PNG storage) and
`numpy.float32` (LZW Float32 TIFF storage):

```python
import numpy as np

from gpkg_tiles import GpkgTiles

with GpkgTiles.create(
    "coverage.gpkg",
    "elevation",
    like=("source.gpkg", "imagery"),
    kind="coverage",
    dtype=np.float32,
    data_null=np.nan,
) as coverage:
    coverage.put(coverage.base_zoom, 0, 0, np.zeros((256, 256), dtype=np.float32))
```

## Development

Run the test suite from the repository root:

```shell
python -m unittest discover
```
