"""PyIceberg access to the lakehouse for gold, training and the app."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.expressions import AlwaysTrue, BooleanExpression

# PyArrow's local FileIO can't parse file:///C:/... warehouse paths on Windows;
# fsspec's can, so the SQL catalog (used by tests) always uses it.
FSSPEC_IO = "pyiceberg.io.fsspec.FsspecFileIO"


@dataclass(frozen=True)
class LakehouseSettings:
    """Where the Iceberg catalog lives. The defaults match the Compose stack."""

    catalog_type: str = "rest"
    uri: str = "http://lakekeeper:8181/catalog"
    warehouse: str = "melee"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "local-01"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "LakehouseSettings":
        defaults = cls()
        return cls(
            catalog_type=env.get("LAKEHOUSE_CATALOG_TYPE", defaults.catalog_type),
            uri=env.get("LAKEKEEPER_URL", defaults.uri),
            warehouse=env.get("LAKEHOUSE_WAREHOUSE", defaults.warehouse),
            s3_endpoint=env.get("S3_ENDPOINT") or None,
            s3_access_key=env.get("S3_ACCESS_KEY") or None,
            s3_secret_key=env.get("S3_SECRET_KEY") or None,
            s3_region=env.get("S3_REGION", defaults.s3_region),
        )


def catalog_properties(settings: LakehouseSettings) -> dict[str, str]:
    """PyIceberg catalog properties. Without explicit S3 keys, Lakekeeper vends them."""
    if settings.catalog_type == "sql":
        return {
            "type": "sql",
            "uri": settings.uri,
            "warehouse": settings.warehouse,
            "py-io-impl": FSSPEC_IO,
        }
    if settings.catalog_type != "rest":
        raise ValueError(f"Unsupported catalog type: {settings.catalog_type}")
    props = {"type": "rest", "uri": settings.uri, "warehouse": settings.warehouse}
    if not settings.s3_access_key:
        props["header.X-Iceberg-Access-Delegation"] = "vended-credentials"
        return props
    if not settings.s3_secret_key:
        raise ValueError("S3_ACCESS_KEY is set but S3_SECRET_KEY is not")
    props.update(
        {
            "s3.access-key-id": settings.s3_access_key,
            "s3.secret-access-key": settings.s3_secret_key,
            "s3.region": settings.s3_region,
        }
    )
    if settings.s3_endpoint:
        props["s3.endpoint"] = settings.s3_endpoint
    return props


def load_lakehouse_catalog(settings: LakehouseSettings | None = None) -> Catalog:
    return load_catalog("melee", **catalog_properties(settings or LakehouseSettings.from_env()))


def read_table(
    catalog: Catalog,
    identifier: str,
    columns: tuple[str, ...],
    row_filter: BooleanExpression = AlwaysTrue(),
    snapshot_id: int | None = None,
) -> pd.DataFrame:
    """Selected columns of an Iceberg table as pandas, optionally pinned to a snapshot."""
    scan = catalog.load_table(identifier).scan(
        row_filter=row_filter, selected_fields=columns, snapshot_id=snapshot_id
    )
    return scan.to_pandas()


def current_snapshot_id(catalog: Catalog, identifier: str) -> int | None:
    snapshot = catalog.load_table(identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id
