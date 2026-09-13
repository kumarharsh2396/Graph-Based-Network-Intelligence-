"""Source-checkout bootstrap for the src-layout package."""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "gbni"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

from .data_pipeline import build_leg_table, load_raw_data  # noqa: E402,F401

__all__ = ["build_leg_table", "load_raw_data"]

