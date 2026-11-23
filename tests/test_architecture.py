"""Structure rules, checked on the source text (no imports, no mocks).

A failure means code took a shortcut past a layer. Move the code into the right layer
instead of loosening a rule; a justified exception goes into the lists below with its reason.
"""
import re
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"

# Files allowed to know the storage layout, with the reason.
LAYOUT_OWNERS = {
    "common/store.py": "the feature store repository: the one place that knows paths",
    "common/storage.py": "the adapter; its docstring shows an example path",
}
PATH_LITERAL = re.compile(r"""["'](raw|clean|features|issued|logs)/""")
STORAGE_READS = re.compile(r"\.(read_parquet|read_json|read_bytes)\(")

# Pure training logic: typed data in, typed result out. No storage, config, tracking or model library.
PURE_TRAINING = ["split.py", "variants.py", "metrics.py", "policy.py"]
FORBIDDEN_IN_PURE = ["common.storage", "common.store", "common.config", "mlflow", "lightgbm", "requests", "fsspec"]


def _sources():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), path.read_text()


def test_only_the_repository_knows_storage_paths():
    offenders = [
        f"{rel}:{n}: {line.strip()}"
        for rel, text in _sources() if rel not in LAYOUT_OWNERS
        for n, line in enumerate(text.splitlines(), 1) if PATH_LITERAL.search(line)
    ]
    assert not offenders, "Storage paths outside common/store.py:\n" + "\n".join(offenders)


def test_only_the_repository_reads_storage():
    offenders = [
        f"{rel}:{n}: {line.strip()}"
        for rel, text in _sources() if rel not in LAYOUT_OWNERS
        for n, line in enumerate(text.splitlines(), 1) if STORAGE_READS.search(line)
    ]
    assert not offenders, "Read stored data through common.store.FeatureStore:\n" + "\n".join(offenders)


def test_training_domain_has_no_io():
    offenders = []
    for name in PURE_TRAINING:
        text = (SRC / "training" / name).read_text()
        for mod in FORBIDDEN_IN_PURE:
            if re.search(rf"^\s*(import|from)\s+{re.escape(mod)}\b", text, re.MULTILINE):
                offenders.append(f"training/{name} imports {mod}")
    assert not offenders, "\n".join(offenders)
