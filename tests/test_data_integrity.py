"""Check data artifact verification before downloading or restoring payloads."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('data_download',ROOT/'scripts/download_data.py')
DOWNLOAD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


def release(tmp_path, name='tables/queries.parquet'):
    path=tmp_path/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'frozen table fixture')
    record=dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    data=dict(schema='mutabench-data-release-v1',manifests=[],tables=[record],archives=[])
    (tmp_path/'RELEASE.json').write_text(json.dumps(data))
    return path,data


def test_checks_frozen_artifact_bytes(tmp_path):
    _,data=release(tmp_path)
    assert DOWNLOAD.verify_artifacts(tmp_path)==data


def test_rejects_tampered_file_even_with_same_size(tmp_path):
    path,_=release(tmp_path)
    path.write_bytes(b'x'*path.stat().st_size)
    with pytest.raises(ValueError,match='checksum mismatch'):
        DOWNLOAD.verify_artifacts(tmp_path)


def test_rejects_artifact_path_traversal(tmp_path):
    _,data=release(tmp_path)
    data['tables'][0]['path']='../outside'
    (tmp_path/'RELEASE.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Unsafe artifact path'):
        DOWNLOAD.verify_artifacts(tmp_path)
