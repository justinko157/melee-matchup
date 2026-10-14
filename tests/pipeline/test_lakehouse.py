import pyarrow as pa
import pytest

from pipeline.lakehouse import (
    LakehouseSettings,
    catalog_properties,
    current_snapshot_id,
    read_table,
)


def test_rest_catalog_requests_vended_credentials():
    props = catalog_properties(LakehouseSettings())
    assert props == {
        "type": "rest",
        "uri": "http://lakekeeper:8181/catalog",
        "warehouse": "melee",
        "header.X-Iceberg-Access-Delegation": "vended-credentials",
    }


def test_explicit_s3_credentials_disable_vending():
    props = catalog_properties(
        LakehouseSettings(
            s3_endpoint="http://localhost:9000", s3_access_key="key", s3_secret_key="secret"
        )
    )
    assert "header.X-Iceberg-Access-Delegation" not in props
    assert props["s3.endpoint"] == "http://localhost:9000"
    assert props["s3.access-key-id"] == "key"
    assert props["s3.secret-access-key"] == "secret"
    assert props["s3.region"] == "local-01"


def test_explicit_credentials_need_a_secret():
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        catalog_properties(LakehouseSettings(s3_access_key="key"))


def test_settings_from_env_defaults():
    assert LakehouseSettings.from_env({}) == LakehouseSettings()


def test_settings_from_env_overrides():
    settings = LakehouseSettings.from_env(
        {
            "LAKEKEEPER_URL": "http://catalog/x",
            "LAKEHOUSE_WAREHOUSE": "w",
            "S3_ENDPOINT": "http://s3",
            "S3_ACCESS_KEY": "a",
            "S3_SECRET_KEY": "b",
        }
    )
    assert settings.uri == "http://catalog/x"
    assert settings.warehouse == "w"
    assert (settings.s3_endpoint, settings.s3_access_key, settings.s3_secret_key) == (
        "http://s3",
        "a",
        "b",
    )


def test_sql_catalog_uses_fsspec_io(sql_settings):
    props = catalog_properties(sql_settings)
    assert props["type"] == "sql"
    assert props["py-io-impl"] == "pyiceberg.io.fsspec.FsspecFileIO"


def test_sql_catalog_round_trip(sql_catalog):
    sql_catalog.create_namespace_if_not_exists("demo")
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    table = sql_catalog.create_table("demo.t", schema=schema)
    assert current_snapshot_id(sql_catalog, "demo.t") is None
    table.append(pa.table({"id": [1, 2], "name": ["a", "b"]}, schema=schema))
    frame = read_table(sql_catalog, "demo.t", ("id",))
    assert sorted(frame["id"].tolist()) == [1, 2]
    assert list(frame.columns) == ["id"]
    assert current_snapshot_id(sql_catalog, "demo.t") is not None


def test_settings_repr_hides_the_secret_key():
    settings = LakehouseSettings(s3_access_key="k", s3_secret_key="secret-value")
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in str(settings)
