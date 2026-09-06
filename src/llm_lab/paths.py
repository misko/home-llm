"""Filesystem layout for the Git control plane and model data plane."""

from __future__ import annotations

import os
import hashlib
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from .errors import StoragePolicyError


DATA_DIRECTORIES = (
    "upstream/hf/hub",
    "upstream/hf/xet",
    "upstream/hf/assets",
    "blobs/sha256",
    "manifests",
    "views",
    "datasets",
    "runs",
    "registry",
    "results",
    "cache",
    "work/download",
    "work/convert",
    "work/verify",
    "quarantine",
    "state",
    "logs",
)

DATA_ROOT_SENTINEL = ".llm-lab-root"
_SENTINEL_CONTENT = b'{"owner":"llm-lab","schema_version":1}\n'


def lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute path without hiding symbolic-link components."""

    return Path(os.path.abspath(Path(path).expanduser()))


def ensure_safe_parent_directory(path: str | Path, *, purpose: str) -> Path:
    """Securely create/validate a file's parent without following symlinks.

    The returned path is lexical rather than resolved: resolving would turn a
    planted symlink into an apparently legitimate path.  Each parent component
    is opened with ``O_NOFOLLOW`` and retained until its child is opened, which
    also avoids a check-then-create seam while constructing the directory tree.
    """

    target = lexical_absolute_path(path)
    _, parent_fd = open_safe_directory(
        target.parent,
        purpose=f"{purpose} parent",
        create=True,
    )
    os.close(parent_fd)
    return target


def open_private_regular_file(
    path: str | Path,
    flags: int,
    *,
    purpose: str,
    mode: int = 0o600,
) -> tuple[Path, int, os.stat_result]:
    """Open one private regular file and bind its name to the opened inode."""

    target = ensure_safe_parent_directory(path, purpose=purpose)
    secure_flags = flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, secure_flags, mode)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise StoragePolicyError(f"unsafe {purpose} path {target}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(target)
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode):
            raise StoragePolicyError(f"unsafe {purpose} path {target}: not a regular file")
        if opened.st_nlink != 1 or named.st_nlink != 1:
            raise StoragePolicyError(
                f"unsafe {purpose} path {target}: hard-linked files are not allowed"
            )
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise StoragePolicyError(
                f"unsafe {purpose} path {target}: path changed while it was opened"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return target, descriptor, opened


def validate_private_regular_file_if_present(
    path: str | Path,
    *,
    purpose: str,
) -> bool:
    """Reject unsafe optional sidecars; return whether a safe file exists."""

    try:
        _, descriptor, _ = open_private_regular_file(
            path,
            os.O_RDONLY | os.O_NONBLOCK,
            purpose=purpose,
        )
    except FileNotFoundError:
        return False
    os.close(descriptor)
    return True


# Backward-compatible private alias for the local call sites below.
_lexical_absolute = lexical_absolute_path


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_managed_directory(name: str | Path, *, dir_fd: int | None = None) -> int:
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=dir_fd)
    except PermissionError as first_error:
        # systemd PrivateTmp may deny a read descriptor for the namespace root
        # while still allowing an O_PATH traversal descriptor. Child managed
        # directories are opened normally, so durability fsyncs remain valid.
        if dir_fd is not None or Path(name) != Path(os.sep) or not hasattr(os, "O_PATH"):
            raise StoragePolicyError(
                f"managed data path is not a safe real directory: {name}: "
                f"{first_error}"
            ) from first_error
        try:
            descriptor = os.open(
                name,
                os.O_PATH
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise StoragePolicyError(
                f"managed data path is not a safe real directory: {name}: {exc}"
            ) from exc
    except OSError as exc:
        raise StoragePolicyError(
            f"managed data path is not a safe real directory: {name}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise StoragePolicyError(
                f"managed data path is not a directory: {name}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_safe_directory(
    path: str | Path,
    *,
    purpose: str,
    create: bool = False,
) -> tuple[Path, int]:
    """Open an absolute directory through component-wise no-follow traversal."""

    target = lexical_absolute_path(path)
    anchor = Path(target.anchor or os.sep)
    current_fd = _open_managed_directory(anchor)
    try:
        for part in target.relative_to(anchor).parts:
            try:
                child_fd = _open_managed_directory(part, dir_fd=current_fd)
            except StoragePolicyError as first_error:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o750, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise StoragePolicyError(
                        f"could not create safe {purpose} directory {target}: {exc}"
                    ) from exc
                try:
                    child_fd = _open_managed_directory(part, dir_fd=current_fd)
                except StoragePolicyError:
                    raise first_error
            os.close(current_fd)
            current_fd = child_fd
        return target, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def open_regular_file_beneath(
    root: str | Path,
    relative: str | Path,
    flags: int,
    *,
    purpose: str,
    require_immutable: bool = False,
) -> tuple[Path, int, os.stat_result]:
    """Open a file beneath a safe root without traversing any symlink."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise StoragePolicyError(f"unsafe {purpose} relative path: {relative}")
    root_path, current_fd = open_safe_directory(root, purpose=purpose, create=False)
    try:
        for part in relative_path.parts[:-1]:
            child_fd = _open_managed_directory(part, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        secure_flags = (
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(relative_path.name, secure_flags, dir_fd=current_fd)
        except OSError as exc:
            raise StoragePolicyError(
                f"unsafe {purpose} path {root_path / relative_path}: {exc}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            named = os.stat(
                relative_path.name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode):
                raise StoragePolicyError(
                    f"unsafe {purpose} path {root_path / relative_path}: "
                    "not a regular file"
                )
            if opened.st_nlink != 1 or named.st_nlink != 1:
                raise StoragePolicyError(
                    f"unsafe {purpose} path {root_path / relative_path}: "
                    "hard-linked files are not allowed"
                )
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise StoragePolicyError(
                    f"unsafe {purpose} path {root_path / relative_path}: "
                    "path changed while it was opened"
                )
            if require_immutable and stat.S_IMODE(opened.st_mode) & 0o222:
                raise StoragePolicyError(
                    f"unsafe {purpose} path {root_path / relative_path}: file is writable"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return root_path / relative_path, descriptor, opened
    finally:
        os.close(current_fd)


def install_immutable_file_beneath(
    root: str | Path,
    relative: str | Path,
    source: str | Path,
    *,
    expected_sha256: str,
    purpose: str,
) -> Path:
    """Atomically publish reviewed bytes through held no-follow directory FDs."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise StoragePolicyError(f"unsafe {purpose} relative path: {relative}")
    source_path, source_fd, source_stat = open_private_regular_file(
        source,
        os.O_RDONLY,
        purpose=f"{purpose} candidate",
    )
    root_path, current_fd = open_safe_directory(root, purpose=purpose, create=False)
    temporary_name = f".{relative_path.name}.install-{secrets.token_hex(12)}"
    temporary_created = False
    try:
        digest = hashlib.sha256()
        while block := os.read(source_fd, 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise StoragePolicyError(
                f"{purpose} candidate digest does not match reviewed SHA-256"
            )
        os.lseek(source_fd, 0, os.SEEK_SET)
        if not stat.S_ISREG(source_stat.st_mode):
            raise StoragePolicyError(f"{purpose} candidate is not a regular file")

        for part in relative_path.parts[:-1]:
            try:
                child_fd = _open_managed_directory(part, dir_fd=current_fd)
            except StoragePolicyError as first_error:
                try:
                    os.mkdir(part, mode=0o750, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise StoragePolicyError(
                        f"could not create {purpose} directory {part}: {exc}"
                    ) from exc
                try:
                    child_fd = _open_managed_directory(part, dir_fd=current_fd)
                except StoragePolicyError:
                    raise first_error
            os.close(current_fd)
            current_fd = child_fd

        try:
            existing = os.stat(
                relative_path.name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise StoragePolicyError(
                f"unsafe {purpose} destination: {root_path / relative_path}"
            )

        destination_fd = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o500,
            dir_fd=current_fd,
        )
        temporary_created = True
        try:
            while block := os.read(source_fd, 1024 * 1024):
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fchmod(destination_fd, 0o555)
            os.fsync(destination_fd)
            os.lseek(destination_fd, 0, os.SEEK_SET)
            installed_digest = hashlib.sha256()
            while block := os.read(destination_fd, 1024 * 1024):
                installed_digest.update(block)
            if installed_digest.hexdigest() != expected_sha256:
                raise StoragePolicyError(
                    f"{purpose} bytes changed during staged installation"
                )
        finally:
            os.close(destination_fd)
        os.replace(
            temporary_name,
            relative_path.name,
            src_dir_fd=current_fd,
            dst_dir_fd=current_fd,
        )
        temporary_created = False
        os.fsync(current_fd)
        return root_path / relative_path
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=current_fd)
            except FileNotFoundError:
                pass
        os.close(current_fd)
        os.close(source_fd)


def _ensure_relative_directory(root_fd: int, relative: str) -> None:
    """Create one managed directory tree using no-follow ``openat`` steps."""

    current_fd = os.dup(root_fd)
    try:
        for part in Path(relative).parts:
            try:
                child_fd = _open_managed_directory(part, dir_fd=current_fd)
            except StoragePolicyError as first_error:
                try:
                    os.mkdir(part, mode=0o750, dir_fd=current_fd)
                except FileExistsError:
                    # A concurrent creator won.  Re-open and validate below.
                    pass
                except OSError as exc:
                    raise StoragePolicyError(
                        f"could not create managed data directory {relative!r}: {exc}"
                    ) from exc
                try:
                    child_fd = _open_managed_directory(part, dir_fd=current_fd)
                except StoragePolicyError:
                    raise first_error
            os.close(current_fd)
            current_fd = child_fd
    finally:
        os.close(current_fd)


def _verify_or_create_sentinel(root: Path, root_fd: int) -> None:
    """Claim a recognizable data root and reject accidental unrelated roots."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(DATA_ROOT_SENTINEL, flags, dir_fd=root_fd)
    except FileNotFoundError:
        allowed = {Path(relative).parts[0] for relative in DATA_DIRECTORIES}
        unexpected = sorted(entry.name for entry in root.iterdir() if entry.name not in allowed)
        if unexpected:
            raise StoragePolicyError(
                f"refusing to claim non-empty unrelated data root {root}; "
                f"unexpected entries: {unexpected}"
            )
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                DATA_ROOT_SENTINEL, create_flags, 0o600, dir_fd=root_fd
            )
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(_SENTINEL_CONTENT)
                stream.flush()
                os.fsync(stream.fileno())
            return
        except FileExistsError:
            descriptor = os.open(DATA_ROOT_SENTINEL, flags, dir_fd=root_fd)
        except OSError as exc:
            raise StoragePolicyError(
                f"could not create data-root sentinel below {root}: {exc}"
            ) from exc
    except OSError as exc:
        raise StoragePolicyError(
            f"could not safely open data-root sentinel below {root}: {exc}"
        ) from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StoragePolicyError(
                f"data-root sentinel is not a private regular file: "
                f"{root / DATA_ROOT_SENTINEL}"
            )
        if metadata.st_size != len(_SENTINEL_CONTENT):
            raise StoragePolicyError(
                f"data-root sentinel has unexpected content: "
                f"{root / DATA_ROOT_SENTINEL}"
            )
        content = os.read(descriptor, len(_SENTINEL_CONTENT) + 1)
        if content != _SENTINEL_CONTENT:
            raise StoragePolicyError(
                f"data-root sentinel has unexpected content: "
                f"{root / DATA_ROOT_SENTINEL}"
            )
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class LabPaths:
    """Resolved locations used by all components."""

    repo_root: Path
    data_root: Path

    @classmethod
    def discover(
        cls,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> "LabPaths":
        repo = Path(
            repo_root or os.environ.get("LLM_LAB_REPO") or Path.cwd()
        ).expanduser().resolve()
        data = _lexical_absolute(
            data_root
            or os.environ.get("LLM_LAB_DATA")
            or repo / ".llm-lab-data"
        )
        return cls(repo_root=repo, data_root=data)

    @property
    def catalog_root(self) -> Path:
        return self.repo_root / "catalog"

    @property
    def hf_hub_cache(self) -> Path:
        return self.data_root / "upstream/hf/hub"

    @property
    def blob_root(self) -> Path:
        return self.data_root / "blobs/sha256"

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests"

    @property
    def view_root(self) -> Path:
        return self.data_root / "views"

    @property
    def registry_path(self) -> Path:
        return self.data_root / "registry/catalog.sqlite"

    @property
    def results_db_path(self) -> Path:
        return self.data_root / "results/results.duckdb"

    @property
    def active_state_path(self) -> Path:
        return self.data_root / "state/active.json"

    @property
    def gpu_lock_path(self) -> Path:
        return self.data_root / "state/gpu0.lock"

    @property
    def gateway_attestation_key_path(self) -> Path:
        return self.data_root / "state/gateway-attestation.key"

    def initialize(self) -> None:
        root = _lexical_absolute(self.data_root)
        if root != self.data_root:
            raise StoragePolicyError(
                f"data root must be an absolute normalized path: {self.data_root}"
            )
        # Validate/create every component, including ``root`` itself.  A check
        # of only the final component would still follow a symlink planted in
        # one of its ancestors.
        ensure_safe_parent_directory(
            root / ".layout-probe",
            purpose="data root",
        )
        root_fd = _open_managed_directory(root)
        try:
            _verify_or_create_sentinel(root, root_fd)
            for relative in DATA_DIRECTORIES:
                _ensure_relative_directory(root_fd, relative)
        finally:
            os.close(root_fd)
