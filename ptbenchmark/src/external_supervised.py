import importlib
import importlib.util
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


EXTERNAL_SUPERVISED_ROOT = Path(__file__).resolve().parents[2] / "external" / "supervised"


class ExternalModelError(RuntimeError):
    pass


@contextmanager
def _prepend_sys_path(path):
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _require_external_repo(name):
    repo_path = EXTERNAL_SUPERVISED_ROOT / name
    if not repo_path.exists():
        raise ExternalModelError(
            f"External supervised model '{name}' not found at {repo_path}. "
            "Run: git submodule update --init --recursive"
        )
    return repo_path


def _load_module_from_file(module_name, file_path, extra_path=None):
    if not file_path.exists():
        raise ExternalModelError(f"Expected model file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ExternalModelError(f"Cannot import external model module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    if extra_path is None:
        spec.loader.exec_module(module)
    else:
        with _prepend_sys_path(extra_path):
            spec.loader.exec_module(module)
    return module


def _install_basicsr_utils_stub():
    if "basicsr.utils" in sys.modules:
        if not hasattr(sys.modules["basicsr.utils"], "scandir"):
            sys.modules["basicsr.utils"].scandir = _scandir
        if not hasattr(sys.modules["basicsr.utils"], "__path__"):
            sys.modules["basicsr.utils"].__path__ = []
        return

    utils_module = types.ModuleType("basicsr.utils")
    utils_module.__path__ = []

    def get_root_logger(*args, **kwargs):
        return logging.getLogger("basicsr")

    utils_module.get_root_logger = get_root_logger
    utils_module.scandir = _scandir
    sys.modules["basicsr.utils"] = utils_module


def _install_nafnet_package_stubs(repo_path):
    packages = {
        "basicsr": repo_path / "basicsr",
        "basicsr.models": repo_path / "basicsr" / "models",
        "basicsr.models.archs": repo_path / "basicsr" / "models" / "archs",
    }
    for package_name, package_path in packages.items():
        module = sys.modules.get(package_name)
        if module is None:
            module = types.ModuleType(package_name)
            sys.modules[package_name] = module
        module.__path__ = [str(package_path)]


def _scandir(dir_path, suffix=None, recursive=False, full_path=False):
    root = Path(dir_path)
    files = root.rglob("*") if recursive else root.iterdir()
    for path in files:
        if not path.is_file():
            continue
        path_str = str(path)
        if suffix is not None and not path_str.endswith(suffix):
            continue
        yield path_str if full_path else str(path.relative_to(root))


def _pad_to_multiple(x, multiple):
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, height, width


class RestormerDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        repo_path = _require_external_repo("Restormer")
        module = _load_module_from_file(
            "ptbenchmark_external_restormer_arch",
            repo_path / "basicsr" / "models" / "archs" / "restormer_arch.py",
        )
        self.model = module.Restormer(
            inp_channels=1,
            out_channels=1,
            dim=24,
            num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=1,
            heads=[1, 2, 4, 8],
        )

    def forward(self, x):
        padded, height, width = _pad_to_multiple(x, 8)
        out = self.model(padded)
        return out[:, :, :height, :width]


class NAFNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        repo_path = _require_external_repo("NAFNet")
        _install_basicsr_utils_stub()
        _install_nafnet_package_stubs(repo_path)
        with _prepend_sys_path(repo_path):
            module = importlib.import_module("basicsr.models.archs.NAFNet_arch")
        self.model = module.NAFNet(
            img_channel=1,
            width=16,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )

    def forward(self, x):
        return self.model(x)


class HIRDiffDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        repo_path = _require_external_repo("HIRDiff")
        with _prepend_sys_path(repo_path):
            module = importlib.import_module("guided_diffusion.sr3_modules.unet")
        self.model = module.UNet(
            in_channel=1,
            out_channel=1,
            inner_channel=32,
            norm_groups=8,
            channel_mults=(1, 2, 4),
            attn_res=(8,),
            res_blocks=1,
            dropout=0,
            image_size=256,
        )

    def forward(self, x):
        padded, height, width = _pad_to_multiple(x, 4)
        outputs = []
        for sample in padded.split(1, dim=0):
            time = torch.ones(1, device=sample.device, dtype=sample.dtype)
            outputs.append(self.model(sample, time))
        out = torch.cat(outputs, dim=0)
        return out[:, :, :height, :width]


def build_external_supervised_model(name):
    builders = {
        "nafnet": NAFNetDenoiser,
        "hirdiff": HIRDiffDenoiser,
        "restormer": RestormerDenoiser,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown external supervised method: {name}") from exc
