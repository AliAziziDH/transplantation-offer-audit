from os import listdir, path
from pathlib import Path
from re import match
from typing import Iterable

from pandas import DataFrame, read_csv, read_excel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_file_name_and_extension(fname: str) -> tuple[str, str | None]:
    """
    Separate a file name from its extension at the first dot.

    Returns ``(fname, None)`` if there is no dot in ``fname``.
    """
    if '.' not in fname:
        return fname, None
    name, ext = fname.split(sep='.', maxsplit=1)
    return name, ext


def get_data_dir(data_dir: str | Path | None = None) -> str:
    """
    Resolve the data directory relative to ``PROJECT_ROOT``.

    Uses ``data_dir`` if provided; otherwise prefers ``PROJECT_ROOT / 'data'``
    when it exists and contains at least one non-hidden file, falling back to
    ``PROJECT_ROOT / 'sample_data'``.
    """
    if data_dir is not None:
        candidate = Path(data_dir)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate.resolve())

    private_data = PROJECT_ROOT / 'data'
    if private_data.is_dir() and any(not p.name.startswith('.') for p in private_data.iterdir()):
        return str(private_data)
    return str(PROJECT_ROOT / 'sample_data')


def get_df_and_name_from_file(fpath: str | Path) -> tuple[DataFrame, str]:
    fpath_str = str(fpath)
    filename = path.basename(fpath_str)
    _, ext = get_file_name_and_extension(fname=filename)
    if ext and ext.endswith('xlsx'):
        return read_excel(fpath_str), filename
    if ext and ext.endswith('csv'):
        return read_csv(fpath_str, low_memory=False), filename
    raise ValueError(f'Unrecognized extension type: {ext}')


def get_dfs_and_names_from_data_dir(
    fname_regex: str | None = None,
    exclude_regex: str | None = None,
    data_dir: str | Path | None = None,
) -> Iterable[tuple[DataFrame, str]]:
    resolved_dir = get_data_dir(data_dir=data_dir)
    for file in sorted(listdir(resolved_dir)):
        fpath = path.join(resolved_dir, file)
        if not path.isfile(fpath):
            continue
        if fname_regex and not match(fname_regex, file):
            continue
        if exclude_regex and match(exclude_regex, file):
            continue
        yield get_df_and_name_from_file(fpath=fpath)
