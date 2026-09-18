"""Small, self-contained API for image and gridded-coverage GeoPackage tiles.

The module deliberately works at the physical GeoPackage-tile level.  It does
not reproject or change a tile matrix set.  ``create(..., like=...)`` copies the
source layer's grid and creates an empty destination layer on that grid.

Pillow and NumPy are optional imports, but are respectively required when an
image or gridded-coverage layer is actually read or written.

Typical use::

    with GpkgTiles.open("source.gpkg", "imagery") as src, \
         GpkgTiles.create("result.gpkg", "processed", like=src,
                          commit_every=500) as dst:
        for z, x, y, image in src.tiles():
            dst.put(z, x, y, transform(image))

Coverage creation accepts ``dtype=numpy.uint16`` (16-bit PNG storage) or
``dtype=numpy.float32`` (LZW Float32 TIFF storage).  Integer coverages are
returned as UInt16 or Int16 when their scale and offset permit an exact integer
result, matching GDAL, and as Float32 otherwise.  Coverage options default to
``"inherit"`` when ``like`` is itself a coverage; passing ``None`` for
``data_null`` explicitly creates a coverage that cannot represent missing
cells.
"""

from __future__ import annotations

import io
import math
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, NamedTuple, TypeAlias

__all__ = [
    "GpkgTiles", "GpkgImageTiles", "GpkgGriddedCoverage",
    "ImageTile", "CoverageTile", "GpkgTilesError", "LayerNotFoundError",
    "LayerExistsError", "InvalidGeoPackageError", "UnsupportedLayerError",
    "InvalidTileError", "CoverageEncodingError", "CoverageNoDataError",
    "MissingDependencyError", "ClosedGeoPackageError",
]


class GpkgTilesError(Exception):
    """Base class for errors raised by this module."""


class LayerNotFoundError(GpkgTilesError):
    """The requested GeoPackage layer does not exist."""


class LayerExistsError(GpkgTilesError):
    """The requested destination layer already exists."""


class InvalidGeoPackageError(GpkgTilesError):
    """A file is not a usable GeoPackage."""


class UnsupportedLayerError(GpkgTilesError):
    """A GeoPackage content type or encoding is unsupported."""


class InvalidTileError(GpkgTilesError):
    """Tile coordinates, dimensions, image type, or array shape are invalid."""


class CoverageEncodingError(GpkgTilesError):
    """Coverage values cannot be represented by the destination encoding."""


class CoverageNoDataError(CoverageEncodingError):
    """Missing values were supplied to a coverage without ``data_null``."""


class MissingDependencyError(GpkgTilesError):
    """An optional dependency required for this operation is unavailable."""


class ClosedGeoPackageError(GpkgTilesError):
    """An operation was attempted after the object was closed."""


if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PillowImage
    ImageData: TypeAlias = PillowImage
    CoverageData: TypeAlias = np.ndarray[Any, Any] | np.ma.MaskedArray[Any, Any]
else:
    ImageData = Any
    CoverageData = Any


class ImageTile(NamedTuple):
    z: int
    x: int
    y: int
    data: ImageData


class CoverageTile(NamedTuple):
    z: int
    x: int
    y: int
    data: CoverageData


ZoomSelector: TypeAlias = Literal["base", "all"] | int
Like: TypeAlias = "GpkgTiles | tuple[os.PathLike[str] | str, str]"

_APP_ID = 0x47504B47
_LEGACY_APP_IDS = {0x47503130, 0x47503131}  # "GP10" and "GP11"
_GPKG_VERSION = 10300
_COVERAGE_EXTENSION = "gpkg_2d_gridded_coverage"
_COVERAGE_DEFINITION = "http://docs.opengeospatial.org/is/17-066r1/17-066r1.html"


def _q(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier or "\x00" in identifier:
        raise ValueError("SQL identifiers must be non-empty strings without NUL bytes")
    return '"' + identifier.replace('"', '""') + '"'


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("NumPy is required for gridded coverage tiles") from exc
    return np


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise MissingDependencyError("Pillow is required for image and coverage tile encoding") from exc
    return Image, UnidentifiedImageError


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
    ).fetchone() is not None


def _columns(db: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row[1] for row in db.execute(f"PRAGMA table_info({_q(table)})"))


class GpkgTiles:
    """Common base class and factory for a single GeoPackage tile layer.

    Use :meth:`open` or :meth:`create`.  Objects own their SQLite connection and
    can be used immediately; a context manager commits on success and rolls back
    uncommitted work on error, then closes the connection.
    """

    kind: Literal["image", "coverage"]

    def __init__(
        self,
        path: os.PathLike[str] | str,
        table: str,
        db: sqlite3.Connection,
        *,
        commit_every: int | None = None,
    ) -> None:
        if commit_every is not None and (isinstance(commit_every, bool) or commit_every <= 0):
            raise ValueError("commit_every must be a positive integer or None")
        self.path = Path(path)
        self.table = table
        self._db: sqlite3.Connection | None = db
        self.commit_every = commit_every
        self._pending = 0

    @classmethod
    def open(
        cls,
        path: os.PathLike[str] | str,
        table: str,
        *,
        commit_every: int | None = None,
    ) -> GpkgTiles:
        """Open an image or gridded-coverage tile layer and detect its subclass."""
        path_obj = Path(path)
        if not path_obj.is_file():
            raise InvalidGeoPackageError(f"GeoPackage file does not exist: {path_obj}")
        db = sqlite3.connect(path_obj)
        db.row_factory = sqlite3.Row
        try:
            cls._validate_package(db, path_obj)
            row = db.execute(
                "SELECT data_type FROM gpkg_contents WHERE table_name=?", (table,)
            ).fetchone()
            if row is None or not _table_exists(db, table):
                raise LayerNotFoundError(f"Tile layer {table!r} was not found in {path_obj}")
            if row[0] == "tiles":
                impl: type[GpkgTiles] = GpkgImageTiles
            elif row[0] == "2d-gridded-coverage":
                impl = GpkgGriddedCoverage
            else:
                raise UnsupportedLayerError(
                    f"Layer {table!r} has unsupported content type {row[0]!r}"
                )
            obj = impl(path_obj, table, db, commit_every=commit_every)
            obj._load_metadata()
            return obj
        except Exception:
            db.close()
            raise

    @classmethod
    def create(
        cls,
        path: os.PathLike[str] | str,
        table: str,
        *,
        like: Like,
        kind: Literal["image", "coverage"] | None = None,
        commit_every: int | None = None,
        dtype: Any = None,
        data_null: float | int | None | Literal["inherit"] = "inherit",
        scale: float | Literal["inherit"] = "inherit",
        offset: float | Literal["inherit"] = "inherit",
        precision: float | None | Literal["inherit"] = "inherit",
        grid_cell_encoding: str = "inherit",
        uom: str | None = "inherit",
        field_name: str | None = "inherit",
        quantity_definition: str | None = "inherit",
        identifier: str | None = None,
        description: str | None = None,
    ) -> GpkgTiles:
        """Create an empty layer, creating the GeoPackage file when necessary.

        ``like`` is an open :class:`GpkgTiles` or ``(path, table)`` pair.  Grid,
        bounds, zoom levels, tile dimensions, CRS, and basic layer metadata are
        copied.  ``kind`` defaults to the source kind.  Coverage-only arguments
        define its layer-wide interpretation; when cloning a coverage, omitted
        values are inherited where appropriate.
        """
        owned_source = False
        if isinstance(like, GpkgTiles):
            source = like
        elif isinstance(like, tuple) and len(like) == 2:
            source = cls.open(like[0], like[1])
            owned_source = True
        else:
            raise TypeError("like must be an open GpkgTiles or a (path, table) pair")
        source._connection()
        target_kind = kind or source.kind
        if target_kind not in ("image", "coverage"):
            raise ValueError("kind must be 'image' or 'coverage'")
        path_obj = Path(path)
        existed = path_obj.exists()
        if existed and not path_obj.is_file():
            if owned_source:
                source.close()
            raise InvalidGeoPackageError(f"Destination is not a file: {path_obj}")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path_obj)
        db.row_factory = sqlite3.Row
        try:
            if existed:
                cls._validate_package(db, path_obj)
            else:
                cls._initialize_package(db)
            if _table_exists(db, table) or db.execute(
                "SELECT 1 FROM gpkg_contents WHERE table_name=?", (table,)
            ).fetchone():
                raise LayerExistsError(f"Layer {table!r} already exists in {path_obj}")
            cls._copy_grid(source, db, table, target_kind, identifier, description)
            if target_kind == "coverage":
                np = _numpy()
                source_coverage = source if isinstance(source, GpkgGriddedCoverage) else None
                if dtype is None and source_coverage is not None:
                    # ``dtype`` selects the on-disk encoding.  A scaled integer
                    # coverage may expose Float32 values, so do not inherit its
                    # decoded dtype here.
                    dtype = (
                        np.uint16
                        if source_coverage._coverage["datatype"] == "integer"
                        else np.float32
                    )
                data_null = (
                    source_coverage.data_null if data_null == "inherit" and source_coverage else
                    None if data_null == "inherit" else data_null
                )
                scale = (
                    source_coverage.scale if scale == "inherit" and source_coverage else
                    1.0 if scale == "inherit" else scale
                )
                offset = (
                    source_coverage.offset if offset == "inherit" and source_coverage else
                    0.0 if offset == "inherit" else offset
                )
                precision = (
                    source_coverage.precision if precision == "inherit" and source_coverage else
                    1.0 if precision == "inherit" else precision
                )
                grid_cell_encoding = (
                    source_coverage.grid_cell_encoding
                    if grid_cell_encoding == "inherit" and source_coverage else
                    "grid-value-is-center" if grid_cell_encoding == "inherit" else grid_cell_encoding
                )
                uom = (
                    source_coverage.uom if uom == "inherit" and source_coverage else
                    None if uom == "inherit" else uom
                )
                field_name = (
                    source_coverage.field_name if field_name == "inherit" and source_coverage else
                    "Height" if field_name == "inherit" else field_name
                )
                quantity_definition = (
                    source_coverage.quantity_definition
                    if quantity_definition == "inherit" and source_coverage else
                    "Height" if quantity_definition == "inherit" else quantity_definition
                )
                dtype = np.dtype(np.float32 if dtype is None else dtype)
                if dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
                    raise ValueError("coverage dtype must be numpy.uint16 or numpy.float32")
                if not math.isfinite(float(scale)) or float(scale) == 0.0:
                    raise ValueError("coverage scale must be finite and non-zero")
                if not math.isfinite(float(offset)):
                    raise ValueError("coverage offset must be finite")
                if dtype == np.dtype(np.float32) and (scale != 1.0 or offset != 0.0):
                    raise ValueError("float32 coverage requires scale=1 and offset=0")
                if precision is not None and (not math.isfinite(precision) or precision <= 0):
                    raise ValueError("precision must be finite and positive, or None")
                if data_null is not None and not math.isfinite(float(data_null)):
                    raise ValueError("data_null must be finite or None")
                if dtype == np.dtype(np.uint16) and data_null is not None:
                    if float(data_null) != int(data_null) or not 0 <= int(data_null) <= 65535:
                        raise ValueError("uint16 data_null must be an integer in 0..65535")
                if dtype == np.dtype(np.float32) and data_null is not None:
                    normalized_null = np.float32(data_null)
                    if not np.isfinite(normalized_null):
                        raise ValueError("float32 data_null is outside the finite Float32 range")
                    data_null = float(normalized_null)
                cls._create_coverage_metadata(
                    db, table, "integer" if dtype == np.dtype(np.uint16) else "float",
                    float(scale), float(offset), precision,
                    None if data_null is None else float(data_null), grid_cell_encoding,
                    uom, field_name, quantity_definition,
                )
            db.commit()
            impl = GpkgGriddedCoverage if target_kind == "coverage" else GpkgImageTiles
            obj = impl(path_obj, table, db, commit_every=commit_every)
            obj._load_metadata()
            return obj
        except Exception:
            db.rollback()
            db.close()
            raise
        finally:
            if owned_source:
                source.close()

    @staticmethod
    def _validate_package(db: sqlite3.Connection, path: Path) -> None:
        try:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        except sqlite3.DatabaseError as exc:
            raise InvalidGeoPackageError(f"Not a SQLite database: {path}") from exc
        required = {"gpkg_spatial_ref_sys", "gpkg_contents", "gpkg_tile_matrix_set", "gpkg_tile_matrix"}
        if not required.issubset(tables):
            raise InvalidGeoPackageError(f"Missing required GeoPackage tables in {path}")
        application_id = int(db.execute("PRAGMA application_id").fetchone()[0])
        if application_id not in {_APP_ID, *_LEGACY_APP_IDS}:
            raise InvalidGeoPackageError(f"SQLite application_id does not identify a GeoPackage: {path}")

    @staticmethod
    def _initialize_package(db: sqlite3.Connection) -> None:
        db.execute(f"PRAGMA application_id={_APP_ID}")
        db.execute(f"PRAGMA user_version={_GPKG_VERSION}")
        db.executescript("""
            CREATE TABLE gpkg_spatial_ref_sys (
              srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY,
              organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
              definition TEXT NOT NULL, description TEXT);
            CREATE TABLE gpkg_contents (
              table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
              identifier TEXT UNIQUE, description TEXT DEFAULT '',
              last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER,
              FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
            CREATE TABLE gpkg_tile_matrix_set (
              table_name TEXT NOT NULL PRIMARY KEY, srs_id INTEGER NOT NULL,
              min_x DOUBLE NOT NULL, min_y DOUBLE NOT NULL,
              max_x DOUBLE NOT NULL, max_y DOUBLE NOT NULL,
              FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id),
              FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name));
            CREATE TABLE gpkg_tile_matrix (
              table_name TEXT NOT NULL, zoom_level INTEGER NOT NULL,
              matrix_width INTEGER NOT NULL, matrix_height INTEGER NOT NULL,
              tile_width INTEGER NOT NULL, tile_height INTEGER NOT NULL,
              pixel_x_size DOUBLE NOT NULL, pixel_y_size DOUBLE NOT NULL,
              CONSTRAINT pk_ttm PRIMARY KEY (table_name, zoom_level),
              FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name));
            INSERT INTO gpkg_spatial_ref_sys VALUES
              ('Undefined Cartesian',-1,'NONE',-1,'undefined','undefined Cartesian coordinate reference system'),
              ('Undefined geographic',0,'NONE',0,'undefined','undefined geographic coordinate reference system'),
              ('WGS 84 geodetic',4326,'EPSG',4326,
               'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
               'longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid');
        """)

    @staticmethod
    def _copy_grid(
        source: "GpkgTiles", db: sqlite3.Connection, table: str,
        kind: str, identifier: str | None, description: str | None,
    ) -> None:
        sdb = source._connection()
        content = sdb.execute("SELECT * FROM gpkg_contents WHERE table_name=?", (source.table,)).fetchone()
        matrix_set = sdb.execute(
            "SELECT * FROM gpkg_tile_matrix_set WHERE table_name=?", (source.table,)
        ).fetchone()
        matrices = sdb.execute(
            "SELECT * FROM gpkg_tile_matrix WHERE table_name=? ORDER BY zoom_level", (source.table,)
        ).fetchall()
        if content is None or matrix_set is None or not matrices:
            raise InvalidGeoPackageError(f"Layer {source.table!r} has incomplete tile metadata")
        srs_id = int(matrix_set["srs_id"])
        src_cols, dst_cols = _columns(sdb, "gpkg_spatial_ref_sys"), _columns(db, "gpkg_spatial_ref_sys")
        common = tuple(c for c in src_cols if c in dst_cols)
        srs = sdb.execute(
            f"SELECT {','.join(map(_q, common))} FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs_id,)
        ).fetchone()
        if srs is not None:
            db.execute(
                f"INSERT OR IGNORE INTO gpkg_spatial_ref_sys ({','.join(map(_q, common))}) VALUES ({','.join('?' for _ in common)})",
                tuple(srs),
            )
        db.execute(f"""CREATE TABLE {_q(table)} (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            zoom_level INTEGER NOT NULL, tile_column INTEGER NOT NULL,
            tile_row INTEGER NOT NULL, tile_data BLOB NOT NULL,
            UNIQUE (zoom_level, tile_column, tile_row))""")
        db.execute(
            """INSERT INTO gpkg_contents
               (table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id)
               VALUES (?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?,?,?,?,?)""",
            (table, "tiles" if kind == "image" else "2d-gridded-coverage",
             identifier if identifier is not None else table,
             description if description is not None else content["description"],
             content["min_x"], content["min_y"], content["max_x"], content["max_y"], content["srs_id"]),
        )
        db.execute(
            "INSERT INTO gpkg_tile_matrix_set VALUES (?,?,?,?,?,?)",
            (table, srs_id, matrix_set["min_x"], matrix_set["min_y"], matrix_set["max_x"], matrix_set["max_y"]),
        )
        db.executemany(
            "INSERT INTO gpkg_tile_matrix VALUES (?,?,?,?,?,?,?,?)",
            [(table, r["zoom_level"], r["matrix_width"], r["matrix_height"],
              r["tile_width"], r["tile_height"], r["pixel_x_size"], r["pixel_y_size"]) for r in matrices],
        )

    @staticmethod
    def _create_coverage_metadata(
        db: sqlite3.Connection, table: str, datatype: str, scale: float, offset: float,
        precision: float | None, data_null: float | None, grid_cell_encoding: str,
        uom: str | None, field_name: str | None, quantity_definition: str | None,
    ) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS gpkg_extensions (
              table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
              definition TEXT NOT NULL, scope TEXT NOT NULL,
              UNIQUE (table_name, column_name, extension_name));
            CREATE TABLE IF NOT EXISTS gpkg_2d_gridded_coverage_ancillary (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tile_matrix_set_name TEXT NOT NULL UNIQUE,
              datatype TEXT NOT NULL DEFAULT 'integer', scale REAL NOT NULL DEFAULT 1.0,
              offset REAL NOT NULL DEFAULT 0.0, precision REAL DEFAULT 1.0,
              data_null REAL, grid_cell_encoding TEXT DEFAULT 'grid-value-is-center',
              uom TEXT, field_name TEXT DEFAULT 'Height', quantity_definition TEXT DEFAULT 'Height',
              FOREIGN KEY (tile_matrix_set_name) REFERENCES gpkg_tile_matrix_set(table_name));
            CREATE TABLE IF NOT EXISTS gpkg_2d_gridded_tile_ancillary (
              id INTEGER PRIMARY KEY AUTOINCREMENT, tpudt_name TEXT NOT NULL,
              tpudt_id INTEGER NOT NULL, scale REAL NOT NULL DEFAULT 1.0,
              offset REAL NOT NULL DEFAULT 0.0, min REAL, max REAL,
              mean REAL, std_dev REAL, UNIQUE (tpudt_name, tpudt_id));
        """)
        for target, column in (
            ("gpkg_2d_gridded_coverage_ancillary", None),
            ("gpkg_2d_gridded_tile_ancillary", None), (table, "tile_data"),
        ):
            exists = db.execute(
                """SELECT 1 FROM gpkg_extensions WHERE table_name=?
                   AND ((column_name IS NULL AND ? IS NULL) OR column_name=?)
                   AND extension_name=?""",
                (target, column, column, _COVERAGE_EXTENSION),
            ).fetchone()
            if exists is None:
                db.execute(
                    "INSERT INTO gpkg_extensions VALUES (?,?,?,?,?)",
                    (target, column, _COVERAGE_EXTENSION, _COVERAGE_DEFINITION, "read-write"),
                )
        db.execute(
            """INSERT INTO gpkg_2d_gridded_coverage_ancillary
               (tile_matrix_set_name,datatype,scale,offset,precision,data_null,
                grid_cell_encoding,uom,field_name,quantity_definition)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (table, datatype, scale, offset, precision, data_null, grid_cell_encoding,
             uom, field_name, quantity_definition),
        )

    def _load_metadata(self) -> None:
        db = self._connection()
        self._matrix = {
            int(r["zoom_level"]): dict(r) for r in db.execute(
                "SELECT * FROM gpkg_tile_matrix WHERE table_name=?", (self.table,)
            )
        }
        if not self._matrix:
            raise InvalidGeoPackageError(f"Layer {self.table!r} has no tile matrices")
        self._contents = dict(db.execute(
            "SELECT * FROM gpkg_contents WHERE table_name=?", (self.table,)
        ).fetchone())
        self._matrix_set = dict(db.execute(
            "SELECT * FROM gpkg_tile_matrix_set WHERE table_name=?", (self.table,)
        ).fetchone())

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            raise ClosedGeoPackageError(f"GeoPackage layer {self.table!r} is closed")
        return self._db

    @property
    def closed(self) -> bool:
        return self._db is None

    @property
    def zooms(self) -> tuple[int, ...]:
        return tuple(sorted(self._matrix))

    @property
    def base_zoom(self) -> int:
        return max(self._matrix)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        m = self._matrix_set
        return (m["min_x"], m["min_y"], m["max_x"], m["max_y"])

    @property
    def srs_id(self) -> int:
        return int(self._matrix_set["srs_id"])

    @property
    def crs(self) -> str:
        row = self._connection().execute(
            "SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?", (self.srs_id,)
        ).fetchone()
        return "" if row is None else str(row[0])

    @property
    def tile_width(self) -> int:
        return int(self._matrix[self.base_zoom]["tile_width"])

    @property
    def tile_height(self) -> int:
        return int(self._matrix[self.base_zoom]["tile_height"])

    def _zoom_values(self, zoom: ZoomSelector) -> tuple[int, ...]:
        if zoom == "base":
            return (self.base_zoom,)
        if zoom == "all":
            return tuple(sorted(self._matrix, reverse=True))
        if isinstance(zoom, bool) or not isinstance(zoom, int):
            raise ValueError("zoom must be 'base', 'all', or an integer")
        if zoom not in self._matrix:
            raise ValueError(f"zoom {zoom} is not defined; available zooms are {self.zooms}")
        return (zoom,)

    def _blob_rows(self, zoom: ZoomSelector) -> Iterator[sqlite3.Row]:
        db = self._connection()
        for z in self._zoom_values(zoom):
            yield from db.execute(
                f"SELECT id,zoom_level,tile_column,tile_row,tile_data FROM {_q(self.table)} "
                "WHERE zoom_level=? ORDER BY tile_row,tile_column", (z,)
            )

    def _get_blob_row(self, z: int, x: int, y: int) -> sqlite3.Row | None:
        self._validate_coords(z, x, y)
        return self._connection().execute(
            f"SELECT id,tile_data FROM {_q(self.table)} WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (z, x, y),
        ).fetchone()

    def _validate_coords(self, z: int, x: int, y: int) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (z, x, y)):
            raise InvalidTileError("z, x, and y must be integers")
        if z not in self._matrix:
            raise InvalidTileError(f"zoom {z} is not defined for layer {self.table!r}")
        m = self._matrix[z]
        if not (0 <= x < m["matrix_width"] and 0 <= y < m["matrix_height"]):
            raise InvalidTileError(
                f"tile ({z}, {x}, {y}) is outside matrix {m['matrix_width']}x{m['matrix_height']}"
            )

    def _put_blob(self, z: int, x: int, y: int, blob: bytes) -> int:
        self._validate_coords(z, x, y)
        db = self._connection()
        db.execute(
            f"""INSERT INTO {_q(self.table)} (zoom_level,tile_column,tile_row,tile_data)
                VALUES (?,?,?,?) ON CONFLICT (zoom_level,tile_column,tile_row)
                DO UPDATE SET tile_data=excluded.tile_data""", (z, x, y, sqlite3.Binary(blob)),
        )
        row = db.execute(
            f"SELECT id FROM {_q(self.table)} WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (z, x, y),
        ).fetchone()
        if row is None:
            raise GpkgTilesError("Tile insert unexpectedly produced no row")
        return int(row[0])

    def _modified(self) -> None:
        self._connection().execute(
            "UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            "WHERE table_name=?", (self.table,),
        )
        self._pending += 1
        if self.commit_every is not None and self._pending >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        self._connection().commit()
        self._pending = 0

    def rollback(self) -> None:
        self._connection().rollback()
        self._pending = 0

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> GpkgTiles:
        self._connection()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        finally:
            self.close()

    def __iter__(self) -> Iterator[Any]:
        return self.tiles()

    def tiles(self, zoom: ZoomSelector = "base") -> Iterator[Any]:
        raise NotImplementedError

    def get(self, z: int, x: int, y: int) -> Any | None:
        raise NotImplementedError

    def put(self, z: int, x: int, y: int, data: Any) -> None:
        raise NotImplementedError

    def build_overviews(
        self,
        levels: Iterable[int] | None = None,
        *,
        resampling: str = "nearest",
        image_format: Literal["webp", "jpeg"] | None = None,
    ) -> None:
        """Rebuild existing coarser zoom levels from the next finer level.

        ``levels`` contains GeoPackage zoom levels, not scale factors.  By
        default all levels below ``base_zoom`` are rebuilt, finest first.
        Existing tiles at rebuilt levels are replaced.  The tile matrix set
        must be aligned, and its pixel-size ratios must be positive.  Image
        overviews default to WebP; ``image_format="jpeg"`` uses JPEG for opaque
        tiles and RGBA PNG for tiles containing transparency.  ``image_format``
        does not apply to gridded coverages.
        """
        targets = sorted(
            (set(self.zooms[:-1]) if levels is None else set(levels)), reverse=True
        )
        if not targets:
            return
        if any(z not in self._matrix or z >= self.base_zoom for z in targets):
            raise ValueError("overview levels must be defined zooms below base_zoom")
        allowed = {"nearest", "average", "bilinear", "bicubic", "lanczos", "mode"}
        method = resampling.lower()
        if method not in allowed:
            raise ValueError(f"resampling must be one of {sorted(allowed)}")
        overview_image_format: str | None = None
        if isinstance(self, GpkgImageTiles):
            overview_image_format = "webp" if image_format is None else image_format.lower()
            if overview_image_format not in {"webp", "jpeg"}:
                raise ValueError("image_format must be 'webp' or 'jpeg'")
        elif image_format is not None:
            raise ValueError("image_format applies only to image tile layers")
        db = self._connection()
        for target_z in targets:
            finer = None
            for candidate in sorted(z for z in self.zooms if z > target_z):
                count = db.execute(
                    f"SELECT COUNT(*) FROM {_q(self.table)} WHERE zoom_level=?", (candidate,)
                ).fetchone()[0]
                if count:
                    finer = candidate
                    break
            if finer is None:
                raise InvalidTileError(f"No populated finer matrix exists for overview zoom {target_z}")
            if isinstance(self, GpkgGriddedCoverage):
                db.execute(
                    "DELETE FROM gpkg_2d_gridded_tile_ancillary WHERE tpudt_name=? AND tpudt_id IN "
                    f"(SELECT id FROM {_q(self.table)} WHERE zoom_level=?)", (self.table, target_z),
                )
            db.execute(f"DELETE FROM {_q(self.table)} WHERE zoom_level=?", (target_z,))
            self._build_level(finer, target_z, method, overview_image_format)

    def _build_level(
        self,
        source_z: int,
        target_z: int,
        method: str,
        image_format: str | None,
    ) -> None:
        sm, tm = self._matrix[source_z], self._matrix[target_z]
        rx = tm["pixel_x_size"] / sm["pixel_x_size"]
        ry = tm["pixel_y_size"] / sm["pixel_y_size"]
        if not (math.isfinite(rx) and math.isfinite(ry) and rx > 0 and ry > 0):
            raise InvalidTileError("Invalid pixel-size ratio in tile matrix metadata")
        source_coords = self._connection().execute(
            f"SELECT tile_column,tile_row FROM {_q(self.table)} WHERE zoom_level=?", (source_z,)
        ).fetchall()
        targets: set[tuple[int, int]] = set()
        for sx, sy in source_coords:
            px0, py0 = sx * sm["tile_width"], sy * sm["tile_height"]
            px1, py1 = px0 + sm["tile_width"], py0 + sm["tile_height"]
            tx0 = int(math.floor(px0 / rx / tm["tile_width"]))
            ty0 = int(math.floor(py0 / ry / tm["tile_height"]))
            tx1 = int(math.ceil(px1 / rx / tm["tile_width"]))
            ty1 = int(math.ceil(py1 / ry / tm["tile_height"]))
            for ty in range(max(0, ty0), min(tm["matrix_height"], ty1)):
                for tx in range(max(0, tx0), min(tm["matrix_width"], tx1)):
                    targets.add((tx, ty))
        for tx, ty in sorted(targets, key=lambda p: (p[1], p[0])):
            result = self._resample_tile(
                source_z, target_z, tx, ty, rx, ry, method, image_format
            )
            if result is not None:
                self.put(target_z, tx, ty, result)

    def _resample_tile(
        self, source_z: int, target_z: int, tx: int, ty: int,
        rx: float, ry: float, method: str, image_format: str | None,
    ) -> Any | None:
        raise NotImplementedError


class GpkgImageTiles(GpkgTiles):
    """A GeoPackage image-tile layer exposed as Pillow images."""

    kind: Literal["image"] = "image"

    def tiles(self, zoom: ZoomSelector = "base") -> Iterator[ImageTile]:
        for row in self._blob_rows(zoom):
            yield ImageTile(row["zoom_level"], row["tile_column"], row["tile_row"],
                            self._decode_image(row["tile_data"]))

    def get(self, z: int, x: int, y: int) -> ImageData | None:
        row = self._get_blob_row(z, x, y)
        return None if row is None else self._decode_image(row["tile_data"])

    @staticmethod
    def _decode_image(blob: bytes) -> ImageData:
        Image, UnidentifiedImageError = _pillow()
        try:
            with Image.open(io.BytesIO(blob)) as opened:
                opened.load()
                fmt = opened.format
                result = opened.copy()
                result.format = fmt
                return result
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidTileError("tile_data is not a supported encoded image") from exc

    def put(self, z: int, x: int, y: int, data: ImageData) -> None:
        Image, _ = _pillow()
        if not isinstance(data, Image.Image):
            raise InvalidTileError("image tile data must be a PIL.Image.Image")
        m = self._matrix.get(z)
        self._validate_coords(z, x, y)
        assert m is not None
        if data.size != (m["tile_width"], m["tile_height"]):
            raise InvalidTileError(
                f"image size {data.size} does not match tile matrix size "
                f"{(m['tile_width'], m['tile_height'])} at zoom {z}"
            )
        blob, fmt = self._encode_image(data)
        db = self._connection()
        # A top-level SAVEPOINT is committed by RELEASE in SQLite.  Ensure the
        # savepoint is nested so put() remains part of the caller's transaction
        # and rollback()/commit_every retain their documented meaning.
        if not db.in_transaction:
            db.execute("BEGIN")
        db.execute("SAVEPOINT gpkg_tiles_put")
        try:
            self._put_blob(z, x, y, blob)
            if fmt == "WEBP":
                self._register_webp()
        except Exception:
            db.execute("ROLLBACK TO gpkg_tiles_put")
            db.execute("RELEASE gpkg_tiles_put")
            raise
        db.execute("RELEASE gpkg_tiles_put")
        self._modified()

    @staticmethod
    def _encode_image(image: ImageData) -> tuple[bytes, str]:
        source_format = (getattr(image, "format", None) or "").upper()
        has_alpha = "A" in image.getbands() or (
            image.mode == "P" and "transparency" in image.info
        )
        if source_format == "PNG":
            fmt = "PNG"
        elif source_format == "JPEG" and not has_alpha:
            fmt = "JPEG"
        elif source_format == "WEBP":
            fmt = "WEBP"
        elif image.mode in ("1", "L", "LA", "P") or len(image.getbands()) <= 2 or has_alpha:
            fmt = "PNG"
        elif image.mode in ("RGB", "CMYK"):
            fmt = "JPEG"
        else:
            fmt = "PNG"
        prepared = image
        if fmt == "JPEG" and image.mode not in ("L", "RGB", "CMYK"):
            prepared = image.convert("RGB")
        elif fmt == "PNG" and image.mode == "F":
            prepared = image.convert("L")
        out = io.BytesIO()
        try:
            prepared.save(out, format=fmt)
        except (KeyError, OSError, ValueError) as exc:
            raise InvalidTileError(f"Pillow could not encode {image.mode!r} image as {fmt}") from exc
        return out.getvalue(), fmt

    def _register_webp(self) -> None:
        db = self._connection()
        db.execute("""CREATE TABLE IF NOT EXISTS gpkg_extensions (
            table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
            definition TEXT NOT NULL, scope TEXT NOT NULL,
            UNIQUE (table_name, column_name, extension_name))""")
        db.execute(
            "INSERT OR IGNORE INTO gpkg_extensions VALUES (?,?,?,?,?)",
            (self.table, "tile_data", "gpkg_webp",
             "http://www.geopackage.org/spec/#extension_tiles_webp", "read-write"),
        )

    def _resample_tile(
        self, source_z: int, target_z: int, tx: int, ty: int,
        rx: float, ry: float, method: str, image_format: str | None,
    ) -> Any | None:
        Image, _ = _pillow()
        sm, tm = self._matrix[source_z], self._matrix[target_z]
        sw = int(math.ceil(tm["tile_width"] * rx))
        sh = int(math.ceil(tm["tile_height"] * ry))
        start_x = int(math.floor(tx * tm["tile_width"] * rx))
        start_y = int(math.floor(ty * tm["tile_height"] * ry))
        canvas = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        sx0, sy0 = start_x // sm["tile_width"], start_y // sm["tile_height"]
        sx1 = int(math.ceil((start_x + sw) / sm["tile_width"]))
        sy1 = int(math.ceil((start_y + sh) / sm["tile_height"]))
        found = False
        for sy in range(max(0, sy0), min(sm["matrix_height"], sy1)):
            for sx in range(max(0, sx0), min(sm["matrix_width"], sx1)):
                tile = self.get(source_z, sx, sy)
                if tile is not None:
                    canvas.alpha_composite(tile.convert("RGBA"),
                        (sx * sm["tile_width"] - start_x, sy * sm["tile_height"] - start_y))
                    found = True
        if not found:
            return None
        filters = {
            "nearest": Image.Resampling.NEAREST, "average": Image.Resampling.BOX,
            "bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS, "mode": Image.Resampling.NEAREST,
        }
        result = canvas.resize((tm["tile_width"], tm["tile_height"]), filters[method])
        if image_format == "webp":
            result.format = "WEBP"
            return result
        if image_format != "jpeg":
            raise AssertionError(f"unexpected overview image format {image_format!r}")
        if result.getchannel("A").getextrema()[0] < 255:
            result.format = "PNG"
            return result
        opaque = result.convert("RGB")
        opaque.format = "JPEG"
        return opaque


class GpkgGriddedCoverage(GpkgTiles):
    """A tiled gridded coverage exposed as decoded NumPy values."""

    kind: Literal["coverage"] = "coverage"

    def _load_metadata(self) -> None:
        super()._load_metadata()
        if not _table_exists(self._connection(), "gpkg_2d_gridded_coverage_ancillary"):
            raise InvalidGeoPackageError("Coverage ancillary table is missing")
        row = self._connection().execute(
            "SELECT * FROM gpkg_2d_gridded_coverage_ancillary WHERE tile_matrix_set_name=?",
            (self.table,),
        ).fetchone()
        if row is None:
            raise InvalidGeoPackageError(f"Coverage metadata is missing for {self.table!r}")
        self._coverage = dict(row)
        if self._coverage["datatype"] not in ("integer", "float"):
            raise UnsupportedLayerError(f"Unsupported coverage datatype {self._coverage['datatype']!r}")
        if self._coverage["datatype"] == "float" and (
            self._coverage["scale"] != 1.0 or self._coverage["offset"] != 0.0
        ):
            raise InvalidGeoPackageError("Float coverage must use coverage scale=1 and offset=0")

    @property
    def dtype(self) -> Any:
        """NumPy dtype returned by :meth:`get` and :meth:`tiles`.

        As in GDAL, integer storage is exposed as UInt16 or Int16 only when
        every scale and offset preserves that type.  Otherwise decoded values
        are exposed as Float32.
        """
        return self._decoded_dtype()

    def _decoded_dtype(self) -> Any:
        np = _numpy()
        if self._coverage["datatype"] == "float":
            return np.dtype(np.float32)

        scale = float(self._coverage["scale"])
        offset = float(self._coverage["offset"])
        data_null = self.data_null
        if scale == 1.0 and offset == 0.0:
            dtype = np.dtype(np.uint16)
            compatible_offsets = (0.0,)
        elif scale == 1.0 and offset == -32768.0:
            dtype = np.dtype(np.int16)
            compatible_offsets = (0.0, 1.0) if data_null == 65535 else (0.0,)
        elif scale == 1.0 and offset == -32767.0 and data_null == 65535:
            dtype = np.dtype(np.int16)
            compatible_offsets = (0.0,)
        else:
            return np.dtype(np.float32)

        placeholders = ",".join("?" for _ in compatible_offsets)
        incompatible = self._connection().execute(
            f"""SELECT 1 FROM gpkg_2d_gridded_tile_ancillary
                WHERE tpudt_name=?
                  AND (scale != 1.0 OR offset NOT IN ({placeholders}))
                LIMIT 1""",
            (self.table, *compatible_offsets),
        ).fetchone()
        return np.dtype(np.float32) if incompatible else dtype

    @property
    def data_null(self) -> float | int | None:
        value = self._coverage["data_null"]
        if value is None:
            return None
        return int(value) if self._coverage["datatype"] == "integer" else float(value)

    @property
    def scale(self) -> float:
        return float(self._coverage["scale"])

    @property
    def offset(self) -> float:
        return float(self._coverage["offset"])

    @property
    def precision(self) -> float | None:
        value = self._coverage["precision"]
        return None if value is None else float(value)

    @property
    def grid_cell_encoding(self) -> str | None:
        return self._coverage["grid_cell_encoding"]

    @property
    def uom(self) -> str | None:
        return self._coverage["uom"]

    @property
    def field_name(self) -> str | None:
        return self._coverage["field_name"]

    @property
    def quantity_definition(self) -> str | None:
        return self._coverage["quantity_definition"]

    def tiles(self, zoom: ZoomSelector = "base") -> Iterator[CoverageTile]:
        decoded_dtype = self._decoded_dtype()
        for row in self._blob_rows(zoom):
            ancillary = self._tile_ancillary(int(row["id"]))
            yield CoverageTile(row["zoom_level"], row["tile_column"], row["tile_row"],
                               self._decode_coverage(row["tile_data"], ancillary, decoded_dtype))

    def get(self, z: int, x: int, y: int) -> CoverageData | None:
        row = self._get_blob_row(z, x, y)
        if row is None:
            return None
        return self._decode_coverage(
            row["tile_data"], self._tile_ancillary(int(row["id"])), self._decoded_dtype()
        )

    def _tile_ancillary(self, tile_id: int) -> sqlite3.Row:
        row = self._connection().execute(
            "SELECT * FROM gpkg_2d_gridded_tile_ancillary WHERE tpudt_name=? AND tpudt_id=?",
            (self.table, tile_id),
        ).fetchone()
        if row is None:
            raise InvalidGeoPackageError(f"Coverage tile id {tile_id} lacks required ancillary metadata")
        if self._coverage["datatype"] == "float" and (row["scale"] != 1.0 or row["offset"] != 0.0):
            raise InvalidGeoPackageError("Float coverage tile must use scale=1 and offset=0")
        return row

    def _decode_coverage(
        self, blob: bytes, ancillary: sqlite3.Row, decoded_dtype: Any
    ) -> CoverageData:
        np = _numpy()
        Image, UnidentifiedImageError = _pillow()
        try:
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
                image_format = image.format
                image_mode = image.mode
                raw = np.asarray(image).copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidTileError("coverage tile_data is not a valid PNG/TIFF image") from exc
        if raw.ndim != 2:
            raise InvalidTileError("coverage tile must contain exactly one component")
        if self._coverage["datatype"] == "integer":
            if image_format != "PNG" or image_mode not in ("I", "I;16"):
                raise InvalidTileError("integer coverage tile must be an unsigned 16-bit PNG")
            if raw.dtype.kind not in "ui" or raw.min(initial=0) < 0 or raw.max(initial=0) > 65535:
                raise InvalidTileError("integer coverage PNG contains values outside UInt16")
            raw = raw.astype(np.uint16, copy=False)
        else:
            if image_format != "TIFF" or image_mode != "F":
                raise InvalidTileError("float coverage tile must be a single-component Float32 TIFF")
            raw = raw.astype(np.float32, copy=False)
        mask = np.zeros(raw.shape, dtype=bool)
        if self.data_null is not None:
            mask = raw == self.data_null
        decoded_dtype = np.dtype(decoded_dtype)
        integer_result = decoded_dtype in (np.dtype(np.uint16), np.dtype(np.int16))
        if integer_result:
            # Masked storage codes can lie outside the decoded integer range.
            # Replace them before casting; their values remain hidden by mask.
            values = ((raw.astype(np.int64) * int(ancillary["scale"]) +
                       int(ancillary["offset"])) * int(self.scale) + int(self.offset))
            if mask.any():
                values[mask] = 0
            values = values.astype(decoded_dtype, copy=False)
        else:
            values = ((raw.astype(np.float64) * float(ancillary["scale"]) +
                       float(ancillary["offset"])) * self.scale + self.offset)
            values = values.astype(np.float32, copy=False)
        return np.ma.array(values, mask=mask, copy=False) if self.data_null is not None else values

    def put(self, z: int, x: int, y: int, data: CoverageData) -> None:
        np = _numpy()
        self._validate_coords(z, x, y)
        array = np.asanyarray(data)
        if array.ndim != 2:
            raise InvalidTileError("coverage tile data must be a two-dimensional array")
        m = self._matrix[z]
        if array.shape != (m["tile_height"], m["tile_width"]):
            raise InvalidTileError(
                f"array shape {array.shape} does not match tile matrix shape "
                f"{(m['tile_height'], m['tile_width'])} at zoom {z}"
            )
        masked = np.ma.getmaskarray(array).copy() if np.ma.isMaskedArray(array) else np.zeros(array.shape, bool)
        numeric = np.asarray(np.ma.getdata(array))
        try:
            numeric = numeric.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise InvalidTileError("coverage values must be numeric") from exc
        masked |= np.isnan(numeric)
        if np.isinf(numeric[~masked]).any():
            raise CoverageEncodingError("coverage values may not contain positive or negative infinity")
        if masked.any() and self.data_null is None:
            raise CoverageNoDataError(
                "Input contains masked or NaN cells, but this coverage has no data_null value"
            )
        valid = numeric[~masked]
        tile_scale, tile_offset = 1.0, 0.0
        if self._coverage["datatype"] == "float":
            encoded = numeric.astype(np.float32)
            if masked.any():
                encoded[masked] = self.data_null
            if not np.isfinite(encoded).all():
                raise CoverageEncodingError("float32 conversion produced a non-finite stored value")
            if self.data_null is not None and np.any(encoded[~masked] == np.float32(self.data_null)):
                raise CoverageEncodingError(
                    "a valid Float32 value collides with the coverage data_null sentinel; "
                    "mask that cell or choose a different data_null"
                )
            blob = self._encode_array(encoded, "TIFF")
        else:
            encoded, tile_scale, tile_offset = self._quantize_uint16(numeric, masked)
            blob = self._encode_array(encoded, "PNG")
        stats = (None, None, None, None) if valid.size == 0 else (
            float(valid.min()), float(valid.max()), float(valid.mean()), float(valid.std(ddof=0))
        )
        db = self._connection()
        # Keep the atomic tile savepoint inside a transaction owned by the
        # layer; releasing a top-level SQLite savepoint would commit the write.
        if not db.in_transaction:
            db.execute("BEGIN")
        db.execute("SAVEPOINT gpkg_tiles_put")
        try:
            tile_id = self._put_blob(z, x, y, blob)
            db.execute(
                """INSERT INTO gpkg_2d_gridded_tile_ancillary
                   (tpudt_name,tpudt_id,scale,offset,min,max,mean,std_dev)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(tpudt_name,tpudt_id) DO UPDATE SET
                     scale=excluded.scale, offset=excluded.offset, min=excluded.min,
                     max=excluded.max, mean=excluded.mean, std_dev=excluded.std_dev""",
                (self.table, tile_id, tile_scale, tile_offset, *stats),
            )
        except Exception:
            db.execute("ROLLBACK TO gpkg_tiles_put")
            db.execute("RELEASE gpkg_tiles_put")
            raise
        db.execute("RELEASE gpkg_tiles_put")
        self._modified()

    @staticmethod
    def _encode_array(array: Any, fmt: str) -> bytes:
        Image, _ = _pillow()
        out = io.BytesIO()
        try:
            image = Image.fromarray(array)
            if fmt == "TIFF":
                image.save(out, format="TIFF", compression="tiff_lzw")
            else:
                image.save(out, format="PNG")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise CoverageEncodingError(f"Pillow could not encode the {fmt} coverage tile") from exc
        return out.getvalue()

    def _quantize_uint16(self, numeric: Any, mask: Any) -> tuple[Any, float, float]:
        np = _numpy()
        null = self.data_null
        if null is None:
            code_min, code_max = 0, 65535
        else:
            n = int(null)
            lower, upper = n, 65535 - n
            if upper > lower:
                code_min, code_max = n + 1, 65535
            else:
                code_min, code_max = 0, n - 1
        if code_max < code_min:
            raise CoverageEncodingError("No UInt16 code remains after reserving data_null")
        valid = numeric[~mask]
        encoded = np.empty(numeric.shape, dtype=np.uint16)
        if valid.size == 0:
            if null is None:
                raise CoverageNoDataError("An entirely missing tile requires data_null")
            encoded.fill(int(null))
            return encoded, 1.0, 0.0
        intermediate = (valid - self.offset) / self.scale
        if not np.isfinite(intermediate).all():
            raise CoverageEncodingError("coverage scale/offset conversion produced non-finite values")
        rounded = np.rint(intermediate)
        direct = (
            np.allclose(intermediate, rounded, rtol=0.0, atol=1e-12)
            and rounded.min() >= code_min and rounded.max() <= code_max
        )
        if direct:
            tile_scale, tile_offset = 1.0, 0.0
            codes = rounded
        else:
            lo, hi = float(intermediate.min()), float(intermediate.max())
            slots = code_max - code_min
            if hi == lo:
                tile_scale = 1.0
                tile_offset = lo - code_min
                codes = np.full(valid.shape, code_min, dtype=np.float64)
            else:
                if slots <= 0:
                    raise CoverageEncodingError("Not enough UInt16 codes to encode this tile")
                minimum_step = 0.0 if self.precision is None else self.precision / abs(self.scale)
                tile_scale = max((hi - lo) / slots, minimum_step)
                tile_offset = lo - code_min * tile_scale
                codes = np.rint((intermediate - tile_offset) / tile_scale)
                codes = np.clip(codes, code_min, code_max)
                reconstructed = (codes * tile_scale + tile_offset) * self.scale + self.offset
                if self.precision is not None and np.max(np.abs(reconstructed - valid)) > self.precision:
                    raise CoverageEncodingError(
                        f"tile cannot be encoded within precision {self.precision!r}"
                    )
        encoded[~mask] = codes.astype(np.uint16)
        if mask.any():
            encoded[mask] = int(null)
        return encoded, float(tile_scale), float(tile_offset)

    def _resample_tile(
        self, source_z: int, target_z: int, tx: int, ty: int,
        rx: float, ry: float, method: str, image_format: str | None,
    ) -> Any | None:
        np = _numpy()
        Image, _ = _pillow()
        sm, tm = self._matrix[source_z], self._matrix[target_z]
        sw = int(math.ceil(tm["tile_width"] * rx))
        sh = int(math.ceil(tm["tile_height"] * ry))
        start_x = int(math.floor(tx * tm["tile_width"] * rx))
        start_y = int(math.floor(ty * tm["tile_height"] * ry))
        values = np.zeros((sh, sw), dtype=np.float32)
        valid = np.zeros((sh, sw), dtype=np.float32)
        sx0, sy0 = start_x // sm["tile_width"], start_y // sm["tile_height"]
        sx1 = int(math.ceil((start_x + sw) / sm["tile_width"]))
        sy1 = int(math.ceil((start_y + sh) / sm["tile_height"]))
        for sy in range(max(0, sy0), min(sm["matrix_height"], sy1)):
            for sx in range(max(0, sx0), min(sm["matrix_width"], sx1)):
                tile = self.get(source_z, sx, sy)
                if tile is None:
                    continue
                arr = np.ma.asarray(tile)
                ox, oy = sx * sm["tile_width"] - start_x, sy * sm["tile_height"] - start_y
                x0, y0 = max(0, ox), max(0, oy)
                x1c, y1c = min(sw, ox + arr.shape[1]), min(sh, oy + arr.shape[0])
                ax0, ay0 = x0 - ox, y0 - oy
                ax1, ay1 = ax0 + (x1c - x0), ay0 + (y1c - y0)
                sub = np.asarray(np.ma.getdata(arr)[ay0:ay1, ax0:ax1], dtype=np.float32)
                ok = ~np.ma.getmaskarray(arr)[ay0:ay1, ax0:ax1]
                values[y0:y1c, x0:x1c][ok] = sub[ok]
                valid[y0:y1c, x0:x1c][ok] = 1.0
        if not valid.any():
            return None
        size = (tm["tile_width"], tm["tile_height"])
        if method == "mode":
            if abs(rx - round(rx)) > 1e-9 or abs(ry - round(ry)) > 1e-9:
                raise InvalidTileError("mode resampling requires integer pixel-size ratios")
            ix, iy = int(round(rx)), int(round(ry))
            out = np.zeros((size[1], size[0]), dtype=np.float32)
            out_valid = np.zeros(out.shape, bool)
            for oy in range(size[1]):
                for ox in range(size[0]):
                    block = values[oy*iy:min((oy+1)*iy, sh), ox*ix:min((ox+1)*ix, sw)]
                    keep = valid[oy*iy:min((oy+1)*iy, sh), ox*ix:min((ox+1)*ix, sw)] > 0
                    if keep.any():
                        vals, counts = np.unique(block[keep], return_counts=True)
                        out[oy, ox] = vals[np.argmax(counts)]
                        out_valid[oy, ox] = True
        else:
            filt = {
                "nearest": Image.Resampling.NEAREST, "average": Image.Resampling.BOX,
                "bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }[method]
            weighted = np.asarray(Image.fromarray(values * valid).resize(size, filt), dtype=np.float32)
            weights = np.asarray(Image.fromarray(valid).resize(size, filt), dtype=np.float32)
            out_valid = weights > 1e-7
            out = np.zeros_like(weighted)
            out[out_valid] = weighted[out_valid] / weights[out_valid]
        if not out_valid.all() and self.data_null is None:
            raise CoverageNoDataError(
                f"overview tile ({target_z}, {tx}, {ty}) has cells without source data, "
                "but the coverage has no data_null"
            )
        return np.ma.array(out, mask=~out_valid) if self.data_null is not None else out
