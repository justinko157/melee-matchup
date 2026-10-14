import re
from pathlib import Path

DATA_PATH = re.compile(r"(?<![\w/])data/(raw|processed|app)\b")


def test_src_package_is_gone():
    assert not Path("src").exists()


def test_no_code_reads_sqlite_or_local_data_files():
    sources = [Path("app.py"), *Path("pipeline").rglob("*.py"), *Path("scripts").rglob("*.py")]
    offenders = [
        str(path)
        for path in sources
        if "sqlite3" in (text := path.read_text(encoding="utf-8")) or DATA_PATH.search(text)
    ]
    assert offenders == []


def test_build_files_no_longer_mention_src():
    for path in ("Makefile", "pyproject.toml", ".github/workflows/ci.yml", "Dockerfile"):
        assert not re.search(r"\bsrc[/.]", Path(path).read_text(encoding="utf-8")), path
