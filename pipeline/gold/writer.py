"""Replace gold tables in one Iceberg commit each."""

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table
from pyiceberg.transforms import YearTransform

from pipeline.gold.tables import GOLD_TABLES


def to_arrow(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)


def ensure_table(catalog: Catalog, identifier: str) -> Table:
    schema, year_column = GOLD_TABLES[identifier]
    catalog.create_namespace_if_not_exists(identifier.split(".")[0])
    if catalog.table_exists(identifier):
        return catalog.load_table(identifier)
    table = catalog.create_table(identifier, schema=schema)
    if year_column:
        with table.update_spec() as update:
            update.add_field(year_column, YearTransform(), f"{year_column}_year")
    return catalog.load_table(identifier)


def replace_table(
    catalog: Catalog, identifier: str, data: pa.Table, properties: dict[str, str]
) -> int | None:
    """Swap the table's contents and properties atomically; return the new snapshot id."""
    table = ensure_table(catalog, identifier)
    with table.transaction() as tx:
        if table.current_snapshot() is None:
            tx.append(data)  # overwrite on a never-written table warns about an empty delete
        else:
            tx.overwrite(data)
        tx.set_properties(properties)
    snapshot = catalog.load_table(identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def write_gold(
    catalog: Catalog, frames: dict[str, pd.DataFrame], properties: dict[str, str]
) -> dict[str, int | None]:
    """Replace every gold table, consuming ``frames``, and set ``properties`` on each.

    Each frame is popped from the dict and released once it is converted to Arrow,
    and set_features (the large one) goes first, so it is gone before the small
    tables are converted. Callers must not hold their own references to the frames.
    """
    snapshots: dict[str, int | None] = {}
    for identifier, (schema, _) in GOLD_TABLES.items():
        frame = frames.pop(identifier)
        data = to_arrow(frame, schema)
        del frame
        snapshots[identifier] = replace_table(catalog, identifier, data, properties)
        del data
    return snapshots
