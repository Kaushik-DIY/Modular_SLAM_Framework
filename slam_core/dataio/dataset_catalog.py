from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from slam_core.dataio.carmen import read_carmen_log
from slam_core.dataio.intel_carmen import read_intel_carmen_log
from slam_core.dataio.lab_carmen import read_lab_carmen_log


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    scan_path: Path
    reader: Callable[[str], list]
    angle_min: float
    angle_max: float
    angle_inc: float
    range_min: float
    range_max: float
    num_beams: int
    beam_stride: int = 1
    imu_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    raw_bag_path: Optional[Path] = None
    has_odom: bool = False
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _profile_freiburg() -> DatasetProfile:
    import numpy as np

    num_beams = 360
    angle_min = -np.pi / 2.0
    angle_max = np.pi / 2.0
    angle_inc = (angle_max - angle_min) / (num_beams - 1)

    base = DATASETS_ROOT / "fr079"
    return DatasetProfile(
        name="fr079",
        scan_path=base / "fr079.clf",
        reader=read_carmen_log,
        angle_min=float(angle_min),
        angle_max=float(angle_max),
        angle_inc=float(angle_inc),
        range_min=0.10,
        range_max=30.0,
        num_beams=num_beams,
        beam_stride=1,
        metadata_path=None,
        readme_path=None,
        raw_bag_path=None,
        has_odom=True,
        initial_pose=(0.0, 0.0, 0.0),
    )


def _profile_intel() -> DatasetProfile:
    import numpy as np

    num_beams = 180
    angle_min = -np.pi / 2.0
    angle_max = np.pi / 2.0
    angle_inc = np.deg2rad(1.0)

    base = DATASETS_ROOT / "intel"
    return DatasetProfile(
        name="intel",
        scan_path=base / "intel.clf",
        reader=read_intel_carmen_log,
        angle_min=float(angle_min),
        angle_max=float(angle_max),
        angle_inc=float(angle_inc),
        range_min=0.10,
        range_max=30.0,
        num_beams=num_beams,
        beam_stride=1,
        metadata_path=None,
        readme_path=None,
        raw_bag_path=None,
        has_odom=True,
        initial_pose=(0.0, 0.0, 0.0),
    )


def _profile_lab_run_2(scan_variant: str = "360") -> DatasetProfile:
    base = DATASETS_ROOT / "lab_run_2"

    if scan_variant == "raw":
        scan_file = "scans.carmen"
        num_beams = 909
        angle_inc = 0.006919807754456997
    elif scan_variant == "360":
        scan_file = "scans_360.carmen"
        num_beams = 360
        angle_inc = 0.01750190942068286
    else:
        raise ValueError(
            f"Unsupported lab_run_2 scan_variant={scan_variant!r}. "
            "Use 'raw' or '360'."
        )

    angle_min = -3.1415927410125732
    angle_max = 3.1415927410125732

    return DatasetProfile(
        name=f"lab_run_2:{scan_variant}",
        scan_path=base / scan_file,
        reader=read_lab_carmen_log,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_inc=angle_inc,
        range_min=0.10000000149011612,
        range_max=16.0,
        num_beams=num_beams,
        beam_stride=1,
        imu_path=base / "imu.csv",
        metadata_path=base / "metadata.json",
        readme_path=base / "README_dataset_format.txt",
        raw_bag_path=base / "lab_run_2.bag",
        has_odom=False,
        initial_pose=(0.0, 0.0, 0.0),
    )


DATASET_NAMES = ("fr079", "intel", "lab_run_2")


def get_dataset_profile(dataset_name: str, *, scan_variant: Optional[str] = None) -> DatasetProfile:
    dataset_name = str(dataset_name)

    if dataset_name == "fr079":
        return _profile_freiburg()
    if dataset_name == "intel":
        return _profile_intel()
    if dataset_name == "lab_run_2":
        return _profile_lab_run_2("360" if scan_variant is None else str(scan_variant))

    raise ValueError(
        f"Unknown dataset_name={dataset_name!r}. Supported values: {', '.join(DATASET_NAMES)}"
    )


def load_dataset_scans(dataset_name: str, *, scan_variant: Optional[str] = None):
    profile = get_dataset_profile(dataset_name, scan_variant=scan_variant)
    scans = profile.reader(str(profile.scan_path))
    return profile, scans