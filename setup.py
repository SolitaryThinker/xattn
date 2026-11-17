import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"


def _load_sm90_manifest_module():
    module_path = CSRC / "flash_sca" / "hopper" / "manifest" / "sm90.py"
    spec = importlib.util.spec_from_file_location(
        "xattn_sm90_build_manifest", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SM90 build manifest: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm90_manifest = _load_sm90_manifest_module()
sm90_head_dims = sm90_manifest.sm90_head_dims
sm90_instantiation_sources = sm90_manifest.sm90_instantiation_sources

TORCH_LIB_DIR = Path(torch.__file__).resolve().parent / "lib"
TORCH_RPATHS = ["$ORIGIN/torch/lib"]
SOURCE_PREFIX_MAP_FLAG = f"-ffile-prefix-map={ROOT}=."
DEFAULT_NVCC_THREADS = 4
MAX_JOBS_CPU_DIVISOR = 2
MAX_JOBS_MEMORY_GB_PER_JOB = 8
SUPPORTED_TARGET_SMS = ("80", "86", "89", "90a", "103a")
SUPPORTED_GENERIC_SMS = ("80", "86", "89")
SUPPORTED_SM90_SMS = ("90a",)
SUPPORTED_SM103_SMS = ("103a",)
# "all" keeps its pre-existing meaning: the family of targets one extension can
# host together (80/86/89 generic plus the Hopper sm_90a family). Blackwell 10.3
# is requested explicitly (103/103a) or discovered from a visible B300 by "auto";
# it is never part of "all", because sm_90a and sm_103a sources cannot share one
# nvcc invocation and mixing them would silently change what "all" builds.
ALL_TARGET_SMS = ("80", "86", "89", "90a")

# The exact nvcc -gencode value for each normalised target. Every declared
# target gets one of these, so a build never silently drops one.
TARGET_GENCODES = {
    "80": "arch=compute_80,code=sm_80",
    "86": "arch=compute_86,code=sm_86",
    "89": "arch=compute_89,code=sm_89",
    "90a": "arch=compute_90a,code=sm_90a",
    "103a": "arch=compute_103a,code=sm_103a",
}
BLACKWELL_GENCODE = "arch=compute_103a,code=sm_103a"
# Torch's TORCH_CUDA_ARCH_LIST spelling of each target. Written out per target
# because a two-digit major (Blackwell 10.x) cannot be derived by splitting the
# target string: "103" -> "1.03" is not an architecture Torch knows.
TORCH_ARCH_BY_TARGET_SM = {
    "80": "8.0",
    "86": "8.6",
    "89": "8.9",
    "90a": "9.0a",
    "103a": "10.3a",
}
# Minimum CUDA toolkit whose nvcc knows arch=compute_103a.
MIN_CUDA_FOR_SM103 = (12, 9)
DEFAULT_TARGET_SM_WITHOUT_VISIBLE_DEVICE = "80"


def _positive_env_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1
    except OSError:
        return os.cpu_count() or 1


def _resolve_host_compiler() -> str:
    requested = os.getenv("CUDAHOSTCXX") or os.getenv("CXX") or "c++"
    compiler = shutil.which(requested)
    if compiler is None:
        raise RuntimeError(f"Cannot resolve CUDA host compiler: {requested}")
    return compiler


def _cxx_standard() -> str:
    torch_version = _version_pair(torch.__version__)
    default = "20" if torch_version >= (2, 12) else "17"
    standard = os.getenv("XATTN_CXX_STANDARD", default)
    if standard not in {"17", "20"}:
        raise RuntimeError("XATTN_CXX_STANDARD must be 17 or 20")
    if standard == "17" and torch_version >= (2, 14):
        raise RuntimeError(
            "PyTorch 2.14+ headers require C++20; "
            "unset XATTN_CXX_STANDARD or set it to 20"
        )
    return f"c++{standard}"


def _version_pair(version: str):
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse version: {version!r}")
    return tuple(map(int, match.groups()))


def _require_supported_toolchain() -> None:
    if _version_pair(torch.__version__) < (2, 10):
        raise RuntimeError("xattn requires PyTorch 2.10 or newer")
    if not torch.version.cuda or _version_pair(torch.version.cuda) < (12, 8):
        raise RuntimeError("xattn requires a CUDA 12.8+ PyTorch build")
    if CUDA_HOME is None:
        raise RuntimeError("xattn requires CUDA Toolkit 12.8+; set CUDA_HOME")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if match is None or _version_pair(match.group(1)) < (12, 8):
        raise RuntimeError("xattn requires CUDA Toolkit 12.8 or newer")
    if _version_pair(match.group(1))[0] != _version_pair(torch.version.cuda)[0]:
        raise RuntimeError(
            "CUDA Toolkit and PyTorch CUDA major versions must match: "
            f"nvcc={match.group(1)}, torch.version.cuda={torch.version.cuda}"
        )


def _torch_cuda_major() -> Optional[int]:
    cuda_version = torch.version.cuda
    if not cuda_version:
        return None
    try:
        return int(cuda_version.split(".", 1)[0])
    except ValueError:
        return None


def _require_sm90_build_supported() -> None:
    cuda_major = _torch_cuda_major()
    if cuda_major is None or cuda_major < 12:
        raise RuntimeError(
            "xattn SM90 builds require a CUDA 12+ PyTorch build. "
            f"Detected torch.version.cuda={torch.version.cuda!r}."
        )


def _cuda_toolkit_version() -> Optional[tuple]:
    """Release of the nvcc that will actually compile the extension."""
    if CUDA_HOME is None:
        return None
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        return None
    try:
        output = subprocess.check_output([str(nvcc), "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+\.\d+)", output)
    if match is None:
        return None
    return _version_pair(match.group(1))


def _require_sm103_build_supported() -> None:
    """A Blackwell target needs an nvcc and a Torch that both know sm_103a.

    Both halves are checked against the real tools, not assumed: the toolkit
    release comes from ``$CUDA_HOME/bin/nvcc --version`` (the compiler that will
    emit the cubin) and the arch string comes from Torch's own flag generator.
    """
    toolkit = _cuda_toolkit_version()
    if toolkit is None:
        raise RuntimeError(
            "Cannot determine the CUDA toolkit version for a Blackwell target; "
            "set CUDA_HOME to a CUDA 12.9+ toolkit."
        )
    if toolkit < MIN_CUDA_FOR_SM103:
        raise RuntimeError(
            "xattn SM103 builds require an nvcc that knows arch=compute_103a "
            f"(CUDA 12.9+); detected nvcc release "
            f"{toolkit[0]}.{toolkit[1]} at {CUDA_HOME}."
        )
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3a"
    try:
        # Torch raises on an arch string it does not know, so this asks Torch
        # itself whether it can emit an sm_103a gencode for this build.
        arch_flags = _get_cuda_arch_flags()
    except Exception as error:  # pragma: no cover - depends on the Torch build
        raise RuntimeError(
            "This PyTorch build cannot emit sm_103a architecture flags; "
            "native Blackwell compilation is unavailable."
        ) from error
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    if f"-gencode={BLACKWELL_GENCODE}" not in arch_flags:
        raise RuntimeError(
            "PyTorch accepted '10.3a' but did not emit "
            f"-gencode={BLACKWELL_GENCODE}: {arch_flags!r}"
        )


def _normalize_target_sm(raw_sm: str) -> str:
    sm = raw_sm.strip().lower()
    for prefix in ("sm_", "sm", "compute_", "compute"):
        if sm.startswith(prefix):
            sm = sm[len(prefix):]
            break
    sm = sm.replace(".", "")
    if sm in ("9", "90"):
        return "90a"
    if sm == "90a":
        return "90a"
    # Blackwell B300 is compute capability 10.3. "103", "10.3", "sm_103a" and
    # "sm_103" all name the same architecture-specific target: this tree has no
    # non-'a' Blackwell FMHA path, so a plain 103 request is served by 103a.
    if sm in ("103", "103a", "10", "100", "100a"):
        if sm in ("10", "100", "100a"):
            raise RuntimeError(
                f"Unsupported XATTN_TARGET_SM entry {raw_sm!r}: only "
                "Blackwell 10.3 (103/103a) is wired in this tree, not 10.0."
            )
        return "103a"
    if sm in SUPPORTED_GENERIC_SMS:
        return sm
    raise RuntimeError(
        "Unsupported XATTN_TARGET_SM entry "
        f"{raw_sm!r}. Supported values are auto, all (the 80/86/89/90a "
        "family), or one or more of 80, 86, 89, 90/90a, and 103/103a. "
        "sm_90a and sm_103a cannot share one extension."
    )


def _gencode_for_target(sm: str) -> str:
    """The nvcc -gencode value for one normalised target."""
    if sm not in TARGET_GENCODES:
        raise RuntimeError(
            f"No -gencode registered for target sm_{sm}; targets are "
            f"{sorted(TARGET_GENCODES)}."
        )
    return TARGET_GENCODES[sm]


def _target_sm_to_torch_arch(sm: str) -> str:
    try:
        return TORCH_ARCH_BY_TARGET_SM[sm]
    except KeyError:  # pragma: no cover - guarded by _normalize_target_sm
        raise RuntimeError(f"No Torch arch string registered for target {sm!r}")


def _visible_target_sms():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return [DEFAULT_TARGET_SM_WITHOUT_VISIBLE_DEVICE]

    target_sms = []
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        if major == 9 and minor == 0:
            target_sm = "90a"
        else:
            target_sm = _normalize_target_sm(f"{major}{minor}")
        if target_sm not in target_sms:
            target_sms.append(target_sm)
    return target_sms


def _parse_target_sms():
    raw_target = os.getenv("XATTN_TARGET_SM")
    if raw_target is None:
        target_sms = _visible_target_sms()
    else:
        target = raw_target.strip().lower()
        if target in ("", "auto"):
            target_sms = _visible_target_sms()
        elif target == "all":
            target_sms = list(ALL_TARGET_SMS)
        else:
            target_sms = []
            for raw_sm in target.replace(";", ",").split(","):
                if raw_sm.strip():
                    target_sm = _normalize_target_sm(raw_sm)
                    if target_sm not in target_sms:
                        target_sms.append(target_sm)
            if not target_sms:
                raise RuntimeError("XATTN_TARGET_SM did not contain any targets")

    if any(sm in SUPPORTED_SM90_SMS for sm in target_sms):
        _require_sm90_build_supported()
    if any(sm in SUPPORTED_SM103_SMS for sm in target_sms):
        _require_sm103_build_supported()
    return target_sms


def _cgroup_memory_headroom_gb() -> Optional[float]:
    """Return the tightest remaining memory budget in the cgroup hierarchy."""
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None

    budgets = []
    for membership in memberships:
        _, controllers, group = membership.split(":", 2)
        v2 = not controllers
        if not v2 and "memory" not in controllers.split(","):
            continue
        for mount in mounts:
            before, after = mount.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if filesystem[0] != ("cgroup2" if v2 else "cgroup"):
                continue
            if not v2 and "memory" not in filesystem[2].split(","):
                continue
            # mountinfo escapes whitespace and backslashes using octal codes.
            def unescape(value):
                for escaped, char in ((r"\040", " "), (r"\011", "\t"),
                                      (r"\012", "\n"), (r"\134", "\\")):
                    value = value.replace(escaped, char)
                return value

            mount_root = Path(unescape(fields[3]))
            mountpoint = Path(unescape(fields[4]))
            try:
                relative = Path(os.path.normpath(group)).relative_to(mount_root)
            except ValueError:
                # A cgroup namespace can report paths relative to its own root.
                relative = Path(os.path.normpath(group).lstrip("/"))
            current = mountpoint / relative
            limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
            usage_name = "memory.current" if v2 else "memory.usage_in_bytes"
            while True:
                try:
                    limit = int((current / limit_name).read_text().strip())
                    usage = int((current / usage_name).read_text().strip())
                    budgets.append(max(0, limit - usage) / (1024 ** 3))
                except (OSError, ValueError):
                    pass  # Includes cgroup v2's unlimited value, "max".
                if current == mountpoint:
                    break
                current = current.parent
    return min(budgets) if budgets else None


def _available_memory_gb() -> Optional[float]:
    budgets = []
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    budgets.append(float(line.split()[1]) / (1024 * 1024))
                    break
    except OSError:
        pass
    cgroup_budget = _cgroup_memory_headroom_gb()
    if cgroup_budget is not None:
        budgets.append(cgroup_budget)
    return min(budgets) if budgets else None


def _default_max_jobs(
    cpu_count: Optional[int] = None,
    available_memory_gb: Optional[float] = None,
) -> int:
    if cpu_count is None:
        cpu_count = _available_cpu_count()
    max_jobs_by_cpu = max(1, cpu_count // MAX_JOBS_CPU_DIVISOR)

    if available_memory_gb is None:
        available_memory_gb = _available_memory_gb()
        if available_memory_gb is None:
            return max_jobs_by_cpu

    max_jobs_by_memory = max(
        1, int(available_memory_gb // MAX_JOBS_MEMORY_GB_PER_JOB)
    )
    return max(1, min(max_jobs_by_cpu, max_jobs_by_memory))


NVCC_THREADS_REQUESTED = _positive_env_integer(
    "XATTN_NVCC_THREADS", DEFAULT_NVCC_THREADS
)
CPU_COUNT = _available_cpu_count()
if os.getenv("MAX_JOBS"):
    BUILD_MAX_JOBS = _positive_env_integer("MAX_JOBS", 1)
else:
    BUILD_MAX_JOBS = _default_max_jobs(
        cpu_count=CPU_COUNT,
    )
NVCC_THREADS = NVCC_THREADS_REQUESTED
CUDA_HOST_COMPILER = _resolve_host_compiler()


class XattnBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.getenv("MAX_JOBS"):
            os.environ["MAX_JOBS"] = str(BUILD_MAX_JOBS)
            print(
                f"Auto set MAX_JOBS={BUILD_MAX_JOBS} for xattn extension build. "
                f"NVCC threads per job: {NVCC_THREADS}. "
                "Override with MAX_JOBS=N if needed."
            )
        super().__init__(*args, **kwargs)


TARGET_SMS = _parse_target_sms()
TARGET_INCLUDES_SM90 = any(
    sm in SUPPORTED_SM90_SMS for sm in TARGET_SMS
)
TARGET_INCLUDES_SM103 = any(
    sm in SUPPORTED_SM103_SMS for sm in TARGET_SMS
)
TARGET_INCLUDES_GENERIC_CUDA = any(
    sm in SUPPORTED_GENERIC_SMS for sm in TARGET_SMS
)
if TARGET_INCLUDES_SM90 and TARGET_INCLUDES_SM103:
    raise RuntimeError(
        "A single xattn extension cannot target both sm_90a and sm_103a: one "
        "nvcc invocation compiles every source for every -gencode, and the "
        "Hopper reader sources are not functional on sm_103a (their GMMA paths "
        "lower to an invalid control path there, and cute gates them on "
        "__CUDA_ARCH_FEAT_SM90_ALL). Build one architecture family per "
        "extension (e.g. XATTN_TARGET_SM=103a for B300)."
    )
BUILD_HAS_SM90 = TARGET_INCLUDES_SM90
BUILD_HAS_SM103 = TARGET_INCLUDES_SM103
BUILD_ONLY_SM90 = (
    BUILD_HAS_SM90 and not TARGET_INCLUDES_GENERIC_CUDA
)
BUILD_ONLY_SM103 = (
    BUILD_HAS_SM103 and not TARGET_INCLUDES_GENERIC_CUDA
)

target_cuda_arch_list = ";".join(
    _target_sm_to_torch_arch(target_sm) for target_sm in TARGET_SMS
)
os.environ["TORCH_CUDA_ARCH_LIST"] = target_cuda_arch_list

print(f"xattn target SMs: {', '.join(TARGET_SMS)}")

FLASH_SCA_HEAD_DIMS = [32, 64, 128, 256]


def _flash_sca_instantiation_sources(kind: str):
    assert kind in {"fwd", "bwd"}
    sources = []
    for dtype in ("fp16", "bf16"):
        for head_dim in FLASH_SCA_HEAD_DIMS:
            for head_dim_v in FLASH_SCA_HEAD_DIMS:
                name = (
                    f"sliding_chunk_attention_{kind}_hdim{head_dim}_"
                    f"vdim{head_dim_v}_{dtype}.cu"
                )
                sources.append(
                    CSRC
                    / "flash_sca"
                    / "cuda"
                    / "instantiations"
                    / name
                )
    return sources


def _flash_sca_sm90_instantiation_sources(kind: str):
    return sm90_instantiation_sources(kind, repo_root=ROOT)


def _flash_swa_sm90_instantiation_sources(kind: str):
    assert kind in {"fwd", "bwd"}
    sources = []
    for dtype in ("fp16", "bf16"):
        for head_dim in sm90_head_dims():
            for head_dim_v in sm90_head_dims():
                name = (
                    f"flash_swa_{kind}_hdim{head_dim}_"
                    f"vdim{head_dim_v}_{dtype}_sm90.cu"
                )
                sources.append(
                    CSRC / "flash_swa" / "hopper" / "instantiations" / name
                )
    return sources


FLASH_SCA_CSRCS = [
    CSRC / "xattn_extension.cc",
    CSRC / "flash_sca" / "sliding_chunk_attention.cc",
    CSRC / "flash_sca" / "sliding_chunk_attention_cuda.cu",
    CSRC / "flash_sca" / "cuda" / "dispatch.cc",
    *_flash_sca_instantiation_sources("fwd"),
    *_flash_sca_instantiation_sources("bwd"),
]

FLASH_SCA_SM90_CSRCS = [
    CSRC / "flash_sca" / "hopper" / "sliding_chunk_attention.cu",
    *_flash_sca_sm90_instantiation_sources("fwd"),
    *_flash_sca_sm90_instantiation_sources("bwd"),
]

CAUSAL_ATTENTION_CSRCS = [
    CSRC / "attention" / "causal_attention.cc",
    CSRC / "causal_flash_attn" / "causal_flash_attn.cc",
    CSRC / "flash_swa" / "flash_swa.cc",
]

FLASH_SWA_SM90_CSRCS = [
    CSRC / "causal_attention" / "launch.cu",
    *_flash_swa_sm90_instantiation_sources("fwd"),
    *_flash_swa_sm90_instantiation_sources("bwd"),
]

SOFTDELTA_CSRCS = [
    CSRC / "softdelta" / "gate.cu",
    CSRC / "softdelta" / "bindings.cc",
]

SOFTDELTA_SM90_CSRCS = [
    CSRC / "softdelta" / "fwd.cu",
    CSRC / "softdelta" / "bwd.cu",
    CSRC / "softdelta" / "preprocess.cu",
    *sorted((CSRC / "softdelta" / "instantiations").glob("*.cu")),
]

ATTENTION_PRECISION_CSRCS = [
    CSRC / "attention" / "precision" / "constant_keys.cu",
    CSRC / "attention" / "precision" / "bindings.cc",
    CSRC / "attention" / "precision" / "zero_sum_repair.cu",
    CSRC / "attention" / "precision" / "preprocess.cu",
    CSRC / "attention" / "precision" / "row_statistics.cu",
    CSRC / "attention" / "precision" / "row_statistics_sm90_launch.cu",
    CSRC / "attention" / "precision" / "probability_repair.cu",
    CSRC / "attention" / "precision" / "deterministic.cu",
    CSRC / "attention" / "precision" / "main.cu",
    CSRC / "attention" / "precision" / "ordered.cu",
    CSRC / "attention" / "precision" / "rectangular.cu",
    CSRC / "attention" / "precision" / "mha_direct.cu",
    *sorted((CSRC / "attention" / "precision" / "instantiations").glob("*.cu")),
]

# Native Blackwell (sm_103a) feature groups. These are the ONLY place native 103
# sources are registered, and a feature macro is defined only when its group is
# complete, so a define can never claim more than is linked:
#
#   XATTN_HAS_NATIVE_READER  - real native reader host + bindings + forward +
#                              basic backward, all linked.
#   XATTN_HAS_NATIVE_PRECISION - additionally the native RowStatistics/Main/
#                              Ordered entry points and the complete portable
#                              helper/bindings set.
#
# XATTN_HAS_SM103 is an ARCHITECTURE macro only: it says a Blackwell gencode was
# requested, and it registers nothing by itself. A group that is partly filled is
# a build error, not a silently reduced feature. A translation unit belongs here
# only once it compiles at -arch=sm_103a AND emits real device code in the
# sm_103a cubin (compiling is not evidence: arch-gated Hopper code compiles to an
# empty or trapping kernel). Hopper-WGMMA units never belong here.
NATIVE_READER_SOURCES = {
    "host": CSRC / "attention" / "native_cuda" / "reader_host.cc",
    "bindings": CSRC / "attention" / "native_cuda" / "bindings.cc",
    "forward_and_backward": CSRC / "attention" / "native_cuda" / "reader_native.cu",
}
NATIVE_PRECISION_SOURCES = {
    "row_statistics": CSRC / "attention" / "precision" / "row_statistics.cu",
    "main_and_ordered": [
        CSRC / "attention" / "precision" / "native" / "adv_bwd_host.cc",
        CSRC / "attention" / "precision" / "native" / "adv_bwd.cu",
        CSRC / "attention" / "precision" / "native" / "adv_bwd_dq.cu",
        CSRC / "attention" / "precision" / "native" / "adv_bwd_dkdv.cu",
    ],
    "helpers": [
        CSRC / "attention" / "precision" / "constant_keys.cu",
        CSRC / "attention" / "precision" / "zero_sum_repair.cu",
        CSRC / "attention" / "precision" / "preprocess.cu",
        CSRC / "attention" / "precision" / "probability_repair.cu",
        CSRC / "attention" / "precision" / "deterministic.cu",
    ],
    "bindings": CSRC / "attention" / "precision" / "bindings.cc",
}


def _flatten_sources(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]
    return [Path(value)]


def _native_group_sources(group, feature):
    """Return the sources of a native feature group, refusing a partial group."""
    missing = [role for role, value in group.items() if not value]
    present = [role for role, value in group.items() if value]
    if missing and present:
        raise RuntimeError(
            f"{feature} is incompletely registered: missing {sorted(missing)} "
            f"while {sorted(present)} are set. Register the whole group or none "
            "of it, so the feature macro cannot claim more than is linked."
        )
    sources = []
    for value in group.values():
        sources.extend(_flatten_sources(value))
    return sources


# Native Blackwell FlashSCA: the host boundary, the bucket dispatch and one
# instantiation per (D bucket, Dv bucket, dtype). XATTN_HAS_NATIVE_SCA is
# defined only when this whole group is linked.
NATIVE_SCA_SOURCES = {
    "host": CSRC / "flash_sca" / "native" / "host.cc",
    # The backward driver: who owns each phase, and the launch of the rest. Lives
    # in this group so that a build links the driver with the bodies it selects
    # between, never one without the other.
    "driver": CSRC / "flash_sca" / "native" / "umma_bwd.cu",
    "dispatch": CSRC / "flash_sca" / "native" / "dispatch.cu",
    "kernels": sorted(
        (CSRC / "flash_sca" / "native" / "instantiations").glob("*.cu")
    ),
}

NATIVE_READER_CSRCS = _native_group_sources(
    NATIVE_READER_SOURCES, "XATTN_HAS_NATIVE_READER"
)
NATIVE_PRECISION_CSRCS = _native_group_sources(
    NATIVE_PRECISION_SOURCES, "XATTN_HAS_NATIVE_PRECISION"
)
NATIVE_SCA_CSRCS = _native_group_sources(
    NATIVE_SCA_SOURCES, "XATTN_HAS_NATIVE_SCA"
)
NATIVE_READER_ENABLED = bool(NATIVE_READER_CSRCS)
NATIVE_SCA_ENABLED = bool(NATIVE_SCA_CSRCS)
NATIVE_PRECISION_ENABLED = bool(NATIVE_PRECISION_CSRCS)
NATIVE_103_CSRCS = (
    *NATIVE_READER_CSRCS,
    *NATIVE_PRECISION_CSRCS,
    *NATIVE_SCA_CSRCS,
)
for _native_source in NATIVE_103_CSRCS:
    if not Path(_native_source).is_file():
        raise RuntimeError(
            f"Native Blackwell source {_native_source} is registered but does "
            "not exist in this checkout. Fix the registration rather than "
            "letting the compiler fail on a missing file."
        )
if NATIVE_PRECISION_ENABLED and not NATIVE_READER_ENABLED:
    raise RuntimeError(
        "XATTN_HAS_NATIVE_PRECISION requires XATTN_HAS_NATIVE_READER: the "
        "native precision entry points need the native reader host and "
        "bindings they extend. Register the NATIVE_READER_SOURCES group too."
    )

if BUILD_ONLY_SM90:
    CSRCS = [
        *ATTENTION_PRECISION_CSRCS,
        CSRC / "xattn_extension.cc",
        CSRC / "flash_sca" / "sliding_chunk_attention.cc",
        *CAUSAL_ATTENTION_CSRCS,
        *FLASH_SCA_SM90_CSRCS,
        *FLASH_SWA_SM90_CSRCS,
        *SOFTDELTA_CSRCS,
        *SOFTDELTA_SM90_CSRCS,
    ]
elif BUILD_ONLY_SM103:
    # Blackwell-only build: no Hopper gencode exists, so no Hopper-TU is legal.
    # The generic tiled reader and the arch-agnostic host/softdelta sources are
    # kept; native Blackwell TUs come only from the completeness-checked groups.
    CSRCS = [
        *FLASH_SCA_CSRCS,
        *CAUSAL_ATTENTION_CSRCS,
        *SOFTDELTA_CSRCS,
        *NATIVE_103_CSRCS,
    ]
elif BUILD_HAS_SM90:
    CSRCS = [
        *ATTENTION_PRECISION_CSRCS,
        *FLASH_SCA_CSRCS,
        *CAUSAL_ATTENTION_CSRCS,
        *FLASH_SCA_SM90_CSRCS,
        *FLASH_SWA_SM90_CSRCS,
        *SOFTDELTA_CSRCS,
        *SOFTDELTA_SM90_CSRCS,
    ]
else:
    CSRCS = [
        *FLASH_SCA_CSRCS,
        *CAUSAL_ATTENTION_CSRCS,
        *SOFTDELTA_CSRCS,
        # Native 103 sources are Blackwell-only: a mixed 103a+generic build
        # compiles them, a generic-only build must not.
        *(NATIVE_103_CSRCS if BUILD_HAS_SM103 else ()),
    ]

CUTLASS_INCLUDE_DIR = Path(
    os.getenv(
        "CUTLASS_INCLUDE_DIR",
        str(CSRC / "cutlass" / "include"),
    )
)
if not CUTLASS_INCLUDE_DIR.is_dir():
    raise RuntimeError(
        "Missing CUTLASS headers at "
        f"{CUTLASS_INCLUDE_DIR}. Set CUTLASS_INCLUDE_DIR to an external "
        "CUTLASS include directory or initialize csrc/cutlass/include "
        "in this checkout."
    )

INCLUDE_DIRS = [
    str(CSRC),
    str(CUTLASS_INCLUDE_DIR),
]

CXX_FLAGS = [
    "-O3",
    f"-std={_cxx_standard()}",
]

if BUILD_HAS_SM90:
    CXX_FLAGS.append("-DXATTN_HAS_SM90")
if BUILD_ONLY_SM90:
    CXX_FLAGS.append("-DXATTN_SM90_ONLY")
if BUILD_HAS_SM103:
    CXX_FLAGS.append("-DXATTN_HAS_SM103")
if BUILD_ONLY_SM103:
    CXX_FLAGS.append("-DXATTN_SM103_ONLY")
# Feature macros say what is actually linked, never what was requested: they are
# derived from the completeness-checked source groups, so they cannot be on
# while their sources are absent, and they are off in a build with no Blackwell.
if BUILD_HAS_SM103:
    if NATIVE_READER_ENABLED:
        CXX_FLAGS.append("-DXATTN_HAS_NATIVE_READER")
    if NATIVE_PRECISION_ENABLED:
        CXX_FLAGS.append("-DXATTN_HAS_NATIVE_PRECISION")
    if NATIVE_SCA_ENABLED:
        CXX_FLAGS.append("-DXATTN_HAS_NATIVE_SCA")

NVCC_FLAGS = [
    "-DNDEBUG",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--threads",
    str(NVCC_THREADS),
    "-ccbin",
    CUDA_HOST_COMPILER,
]

if BUILD_HAS_SM90:
    NVCC_FLAGS.append("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED")

# One -gencode per declared target, always. Passing any -gencode that contains
# "arch" makes torch's arch-flag resolution return nothing, so every declared
# target must be emitted here or it is silently not built (that is how the seed
# built only sm_90a for XATTN_TARGET_SM=80,90a). Nothing is inherited from
# TORCH_CUDA_ARCH_LIST: what is declared is what is compiled.
for _target_sm in TARGET_SMS:
    NVCC_FLAGS.extend(["-gencode", _gencode_for_target(_target_sm)])


def main():
    _require_supported_toolchain()
    setup(
        name="xattn",
        version="1.2.0",
        description="Efficient attention modules for xLLM.",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/ifm-ai/xattn",
        license="Apache-2.0",
        license_files=["LICENSE", "THIRD_PARTY_NOTICES"],
        packages=find_packages(exclude=("tests", "tests.*")),
        ext_modules=[
            CUDAExtension(
                "xattn_cuda",
                [str(path.relative_to(ROOT)) for path in CSRCS],
                include_dirs=INCLUDE_DIRS,
                library_dirs=[str(TORCH_LIB_DIR)],
                extra_compile_args={
                    "cxx": CXX_FLAGS + [SOURCE_PREFIX_MAP_FLAG],
                    "nvcc": CXX_FLAGS
                    + [f"-Xcompiler={SOURCE_PREFIX_MAP_FLAG}"]
                    + NVCC_FLAGS,
                },
                extra_link_args=[
                    f"-Wl,-rpath,{rpath}"
                    for rpath in TORCH_RPATHS
                ],
            )
        ],
        cmdclass={"build_ext": XattnBuildExtension},
        python_requires=">=3.10",
        install_requires=["torch>=2.10"],
        extras_require={"test": ["pytest"]},
    )


if __name__ == "__main__":
    main()
