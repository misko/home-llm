"""Content-addressed artifact storage and integrity operations."""

from __future__ import annotations

import errno
import fnmatch
import json
import os
import re
import shutil
import stat
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

from .errors import CatalogError, IntegrityError, StoragePolicyError
from .hashing import canonical_json_bytes, canonical_sha256, sha256_file, sha256_uri
from .paths import DATA_DIRECTORIES, LabPaths
from .registry import Registry, manifest_identity_sha256, seal_manifest
from .schema import (
    ArtifactFileSelector,
    ArtifactManifest,
    ArtifactSpec,
    LockedFile,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_IMMUTABLE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_IMMUTABLE_DIRECTORY_MODE = (
    stat.S_IRUSR
    | stat.S_IXUSR
    | stat.S_IRGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IXOTH
)
BlobInstallMode = Literal["copy", "hardlink"]


@dataclass(frozen=True, slots=True)
class PromotionResult:
    manifest: ArtifactManifest
    manifest_path: Path
    view_path: Path
    new_blob_count: int
    reused_blob_count: int
    new_bytes: int
    reused_bytes: int

    @property
    def added_bytes(self) -> int:
        return self.new_bytes


@dataclass(frozen=True, slots=True)
class VerificationReport:
    artifact_id: str
    manifest_sha256: str
    file_count: int
    total_logical_bytes: int
    view_verified: bool


@dataclass(frozen=True, slots=True)
class BlobCandidate:
    sha256: str
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class GCReport:
    dry_run: bool
    candidates: tuple[BlobCandidate, ...]
    candidate_bytes: int
    removed: tuple[BlobCandidate, ...]
    removed_bytes: int

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def removed_count(self) -> int:
        return len(self.removed)


@dataclass(frozen=True, slots=True)
class HFPullResult:
    artifact_id: str
    snapshot_path: Path
    resolved_revision: str
    selected_patterns: tuple[str, ...]
    selected_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectedFile:
    logical_path: str
    source_path: Path
    role: Any
    required: bool
    size_bytes: int
    sha256: str


class ArtifactStore:
    """Promote selected model files into an immutable SHA-256 store."""

    def __init__(
        self,
        paths: LabPaths | str | Path,
        *,
        registry: Registry | None = None,
        free_reserve_bytes: int = 0,
        blob_install_mode: BlobInstallMode = "copy",
    ) -> None:
        if isinstance(paths, LabPaths):
            self.paths = paths
        else:
            raw_data_root = Path(paths).expanduser().absolute()
            if raw_data_root.is_symlink():
                raise StoragePolicyError(
                    f"data root must not be a symbolic link: {raw_data_root}"
                )
            self.paths = LabPaths.discover(data_root=raw_data_root)
        if free_reserve_bytes < 0:
            raise ValueError("free_reserve_bytes cannot be negative")
        if blob_install_mode not in {"copy", "hardlink"}:
            raise ValueError("blob_install_mode must be 'copy' or 'hardlink'")
        self.free_reserve_bytes = free_reserve_bytes
        # Copy is intentionally the default.  A hardlink couples permissions
        # and future mutation of the source inode to CAS.  The opt-in mode is
        # useful only when the source is deliberately adopted as immutable.
        self.blob_install_mode = blob_install_mode
        self._lock_path = self.paths.data_root / "state/storage.lock"
        self._assert_managed_layout()
        self.paths.initialize()
        self._assert_managed_layout()
        self._owns_registry = registry is None
        self.registry = registry or Registry(self.paths.registry_path)
        self._thread_lock = threading.RLock()

    def close(self) -> None:
        if self._owns_registry:
            self.registry.close()

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def blob_path(self, sha256: str) -> Path:
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError("invalid lowercase SHA-256 digest")
        return self.paths.blob_root / sha256

    def manifest_path(self, manifest: ArtifactManifest) -> Path:
        if manifest.manifest_sha256 is None:
            raise IntegrityError("manifest must be sealed before choosing its path")
        destination = (
            self.paths.manifest_root
            / manifest.artifact_id
            / f"{manifest.manifest_sha256}.json"
        )
        self._assert_safe_managed_path(destination, self.paths.manifest_root)
        return destination

    def view_path(self, artifact_id: str) -> Path:
        destination = self.paths.view_root / artifact_id
        self._assert_safe_managed_path(destination, self.paths.view_root)
        return destination

    def promote(
        self,
        artifact: ArtifactSpec | Mapping[str, Any],
        resolved_tree: str | Path,
        resolved_revision: str | None = None,
        *,
        alias: str | None = None,
        expected_size_bytes: int | None = None,
        enforce_expected_size: bool = True,
    ) -> PromotionResult:
        """Promote a resolved local tree into CAS, manifest, registry and view.

        Required selectors must match.  Existing blobs are verified before
        reuse.  Files are copied by default; explicit ``blob_install_mode=
        "hardlink"`` adopts the source inode as read-only CAS content and falls
        back to copying if linking fails.  Views are published by a directory
        rename only after every link exists.
        """

        spec = (
            artifact
            if isinstance(artifact, ArtifactSpec)
            else ArtifactSpec.model_validate(artifact)
        )
        self._assert_managed_layout()
        revision = resolved_revision or spec.source.revision
        if not revision or not revision.strip():
            raise ValueError("resolved_revision must be non-empty")

        root = Path(resolved_tree).expanduser().resolve()
        selected = self._select_files(root, spec.files)
        total_logical = sum(item.size_bytes for item in selected)
        expected = (
            expected_size_bytes
            if expected_size_bytes is not None
            else spec.expected_size_bytes
        )
        if expected is not None and expected <= 0:
            raise ValueError("expected_size_bytes must be positive")
        if enforce_expected_size and expected is not None and total_logical != expected:
            raise IntegrityError(
                f"artifact {spec.id!r} selected {total_logical} bytes; "
                f"catalog expected {expected}"
            )

        existing_manifest = self.registry.find_artifact(spec.id)
        created_at = (
            existing_manifest.created_at
            if existing_manifest is not None
            else datetime.now(timezone.utc)
        )
        manifest = self._build_manifest(
            spec,
            selected,
            resolved_revision=revision,
            created_at=created_at,
        )
        if (
            existing_manifest is not None
            and existing_manifest.manifest_sha256 != manifest.manifest_sha256
        ):
            if manifest_identity_sha256(existing_manifest) != (
                manifest_identity_sha256(manifest)
            ):
                raise CatalogError(
                    f"artifact id {spec.id!r} is already registered with different "
                    "immutable content"
                )
            # Preserve the stored legacy digest/path on an idempotent import.
            manifest = existing_manifest

        unique_files: dict[str, _SelectedFile] = {}
        for item in selected:
            unique_files.setdefault(item.sha256, item)

        manifest_path = self.manifest_path(manifest)
        view_path = self.view_path(spec.id)
        new_manifest = False
        new_view = False
        with self._storage_lock():
            new_digests: list[str] = []
            for digest, item in unique_files.items():
                destination = self.blob_path(digest)
                if destination.exists():
                    self._verify_blob(destination, digest, item.size_bytes)
                else:
                    new_digests.append(digest)
            new_bytes = sum(unique_files[digest].size_bytes for digest in new_digests)
            self._ensure_free_space(self.paths.blob_root, new_bytes)

            for digest in new_digests:
                item = unique_files[digest]
                self._install_blob(item.source_path, digest, item.size_bytes)

            try:
                new_manifest = self._write_manifest(manifest, manifest_path)
                new_view = self._create_view(manifest, view_path)
                registered = self.registry.register_artifact(
                    manifest, manifest_path=manifest_path
                )
                if registered.manifest_sha256 != manifest.manifest_sha256:
                    raise IntegrityError(
                        f"registry returned an unexpected manifest for {spec.id!r}"
                    )
            except BaseException:
                # CAS writes are safe orphans and can be reclaimed by GC.  A
                # newly published manifest/view is removed so callers never see
                # a half-registered artifact.
                if new_view and view_path.exists():
                    self._set_view_directory_mode(view_path, 0o750)
                    shutil.rmtree(view_path)
                if new_manifest and manifest_path.exists():
                    manifest_path.unlink()
                    _remove_empty_parent(manifest_path.parent, self.paths.manifest_root)
                raise

        if alias is not None:
            self.registry.set_alias(alias, spec.id)

        reused_bytes = total_logical - new_bytes
        return PromotionResult(
            manifest=manifest,
            manifest_path=manifest_path,
            view_path=view_path,
            new_blob_count=len(new_digests),
            reused_blob_count=len(unique_files) - len(new_digests),
            new_bytes=new_bytes,
            reused_bytes=reused_bytes,
        )

    # Compatibility-friendly, explicit name for callers and scripts.
    promote_tree = promote

    def verify(
        self,
        artifact: str | Path | ArtifactManifest | Mapping[str, Any],
        *,
        verify_view: bool = True,
    ) -> VerificationReport:
        """Re-hash an artifact manifest, all CAS blobs and optionally its view."""

        manifest = self._resolve_manifest(artifact)
        self._assert_managed_layout()
        if manifest.manifest_sha256 is None:
            raise IntegrityError(
                f"artifact {manifest.artifact_id!r} has an unsealed manifest"
            )
        sealed = seal_manifest(manifest)
        self._verify_manifest_file(sealed)
        if len({item.logical_path for item in sealed.files}) != len(sealed.files):
            raise IntegrityError("manifest contains duplicate logical paths")

        expected_total = 0
        for item in sealed.files:
            if item.storage_uri != sha256_uri(item.sha256):
                raise IntegrityError(
                    f"storage URI does not match digest for {item.logical_path!r}"
                )
            blob = self.blob_path(item.sha256)
            self._verify_blob(blob, item.sha256, item.size_bytes)
            expected_total += item.size_bytes

        if expected_total != sealed.total_logical_bytes:
            raise IntegrityError(
                f"manifest total is {sealed.total_logical_bytes}, but files total "
                f"{expected_total}"
            )
        tree_digest = self._tree_digest(sealed.files)
        if tree_digest != sealed.tree_sha256:
            raise IntegrityError(
                f"tree digest mismatch for artifact {sealed.artifact_id!r}"
            )

        if verify_view:
            view = self.view_path(sealed.artifact_id)
            if not self._view_matches(sealed, view):
                raise IntegrityError(
                    f"materialized view is missing, extra, or corrupt: {view}"
                )

        return VerificationReport(
            artifact_id=sealed.artifact_id,
            manifest_sha256=sealed.manifest_sha256,
            file_count=len(sealed.files),
            total_logical_bytes=sealed.total_logical_bytes,
            view_verified=verify_view,
        )

    verify_artifact = verify

    def gc(
        self,
        *,
        dry_run: bool = True,
        keep_artifact_ids: Iterable[str] | None = None,
    ) -> GCReport:
        """Report or remove CAS blobs not referenced by live manifests.

        By default both registered and on-disk manifests are roots.  Passing an
        explicit artifact set is intended for a higher-level retention policy;
        callers are responsible for removing stale manifests/views afterward.
        """

        with self._storage_lock():
            # Roots and candidates must be observed under the same lock used by
            # promotion.  Otherwise a promotion can publish between these two
            # phases and its newly-live blobs can be collected as stale.
            self._assert_managed_layout()
            live = self._live_manifests(keep_artifact_ids)
            referenced = {
                item.sha256 for manifest in live for item in manifest.files
            }
            candidates: list[BlobCandidate] = []
            for path in sorted(self.paths.blob_root.iterdir()):
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise StoragePolicyError(
                        f"CAS blob directory contains a symbolic link: {path}"
                    )
                if not stat.S_ISREG(metadata.st_mode) or not _SHA256_RE.fullmatch(
                    path.name
                ):
                    continue
                if path.name not in referenced:
                    candidates.append(
                        BlobCandidate(path.name, path, metadata.st_size)
                    )

            removed: list[BlobCandidate] = []
            if not dry_run:
                for candidate in candidates:
                    try:
                        candidate.path.unlink()
                    except FileNotFoundError:
                        continue
                    removed.append(candidate)

        return GCReport(
            dry_run=dry_run,
            candidates=tuple(candidates),
            candidate_bytes=sum(item.size_bytes for item in candidates),
            removed=tuple(removed),
            removed_bytes=sum(item.size_bytes for item in removed),
        )

    garbage_collect = gc

    def pull_huggingface(
        self,
        artifact: ArtifactSpec | Mapping[str, Any],
        *,
        token: str | bool | None = None,
        local_files_only: bool = False,
        api: Any | None = None,
        snapshot_download_fn: Callable[..., str] | None = None,
    ) -> HFPullResult:
        """Resolve once, then download selected HF files at the pinned commit."""

        spec = (
            artifact
            if isinstance(artifact, ArtifactSpec)
            else ArtifactSpec.model_validate(artifact)
        )
        self._assert_managed_layout()
        source = spec.source
        if source.provider != "huggingface" or not source.repo_id:
            raise CatalogError(
                f"artifact {spec.id!r} does not have a Hugging Face source"
            )
        if spec.expected_size_bytes is not None:
            self._ensure_free_space(
                self.paths.hf_hub_cache, spec.expected_size_bytes
            )

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:  # pragma: no cover - project dependency
            raise StoragePolicyError(
                "huggingface_hub is required to pull Hugging Face artifacts"
            ) from exc

        downloader = snapshot_download_fn or snapshot_download
        revision = source.revision
        if local_files_only and _HF_COMMIT_RE.fullmatch(revision):
            resolved = revision
        else:
            client = api or HfApi()
            info = client.repo_info(
                repo_id=source.repo_id,
                repo_type=source.repo_type,
                revision=revision,
                token=token,
            )
            resolved = getattr(info, "sha", None)
            if not isinstance(resolved, str) or not _HF_COMMIT_RE.fullmatch(resolved):
                raise IntegrityError(
                    f"Hub did not resolve {source.repo_id}@{revision} to a full commit"
                )

        patterns = tuple(dict.fromkeys(selector.pattern for selector in spec.files))
        snapshot = Path(
            downloader(
                repo_id=source.repo_id,
                repo_type=source.repo_type,
                revision=resolved,
                allow_patterns=list(patterns),
                cache_dir=str(self.paths.hf_hub_cache),
                token=token,
                local_files_only=local_files_only,
            )
        ).resolve()
        selected = self._select_files(snapshot, spec.files)
        actual_size = sum(item.size_bytes for item in selected)
        if (
            spec.expected_size_bytes is not None
            and actual_size != spec.expected_size_bytes
        ):
            raise IntegrityError(
                f"artifact {spec.id!r} downloaded {actual_size} selected bytes; "
                f"catalog expected {spec.expected_size_bytes}"
            )
        return HFPullResult(
            artifact_id=spec.id,
            snapshot_path=snapshot,
            resolved_revision=resolved,
            selected_patterns=patterns,
            selected_files=tuple(item.logical_path for item in selected),
        )

    def pull_and_promote(
        self,
        artifact: ArtifactSpec | Mapping[str, Any],
        *,
        alias: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> PromotionResult:
        pulled = self.pull_huggingface(
            artifact, token=token, local_files_only=local_files_only
        )
        return self.promote(
            artifact,
            pulled.snapshot_path,
            pulled.resolved_revision,
            alias=alias,
        )

    def _select_files(
        self,
        root: Path,
        selectors: Sequence[ArtifactFileSelector],
    ) -> tuple[_SelectedFile, ...]:
        if not root.is_dir():
            raise IntegrityError(f"resolved artifact tree is not a directory: {root}")

        matches_by_path: dict[
            str, tuple[Path, Any, bool, int | None, str | None]
        ] = {}
        for selector in selectors:
            _validate_pattern(selector.pattern)
            matches = sorted(
                path
                for path in root.glob(selector.pattern)
                if path.is_file()
                and artifact_path_matches(
                    path.relative_to(root).as_posix(), selector.pattern
                )
            )
            if selector.required and not matches:
                raise IntegrityError(
                    f"required artifact selector {selector.pattern!r} matched no files "
                    f"under {root}"
                )
            for path in matches:
                try:
                    logical = path.relative_to(root).as_posix()
                except ValueError as exc:  # defensive; glob should stay below root
                    raise IntegrityError(f"selected path escapes artifact tree: {path}") from exc
                previous = matches_by_path.get(logical)
                if previous is not None and previous[1] != selector.role:
                    raise CatalogError(
                        f"artifact file {logical!r} has conflicting roles "
                        f"{previous[1].value!r} and {selector.role.value!r}"
                    )
                required = selector.required or (previous[2] if previous else False)
                expected_size = selector.expected_size_bytes
                expected_sha256 = selector.expected_sha256
                if previous is not None:
                    if (
                        previous[3] is not None
                        and expected_size is not None
                        and previous[3] != expected_size
                    ) or (
                        previous[4] is not None
                        and expected_sha256 is not None
                        and previous[4] != expected_sha256
                    ):
                        raise CatalogError(
                            f"artifact file {logical!r} has conflicting expected content"
                        )
                    expected_size = expected_size or previous[3]
                    expected_sha256 = expected_sha256 or previous[4]
                matches_by_path[logical] = (
                    path,
                    selector.role,
                    required,
                    expected_size,
                    expected_sha256,
                )

        if not matches_by_path:
            raise IntegrityError(f"artifact selectors matched no files under {root}")

        selected: list[_SelectedFile] = []
        for logical, (
            path,
            role,
            required,
            expected_size,
            expected_sha256,
        ) in sorted(matches_by_path.items()):
            source = path.resolve(strict=True)
            before = source.stat()
            if not stat.S_ISREG(before.st_mode):
                raise IntegrityError(f"selected artifact path is not a file: {path}")
            digest = sha256_file(source)
            after = source.stat()
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if before_identity != after_identity:
                raise IntegrityError(f"artifact file changed while hashing: {path}")
            if expected_size is not None and after.st_size != expected_size:
                raise IntegrityError(
                    f"artifact file {logical!r} is {after.st_size} bytes; "
                    f"catalog expected {expected_size}"
                )
            if expected_sha256 is not None and digest != expected_sha256:
                raise IntegrityError(
                    f"artifact file {logical!r} has SHA-256 {digest}; "
                    f"catalog expected {expected_sha256}"
                )
            selected.append(
                _SelectedFile(
                    logical_path=logical,
                    source_path=source,
                    role=role,
                    required=required,
                    size_bytes=after.st_size,
                    sha256=digest,
                )
            )
        return tuple(selected)

    def _build_manifest(
        self,
        spec: ArtifactSpec,
        selected: Sequence[_SelectedFile],
        *,
        resolved_revision: str,
        created_at: datetime,
    ) -> ArtifactManifest:
        locked = tuple(
            LockedFile(
                logical_path=item.logical_path,
                role=item.role,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                storage_uri=sha256_uri(item.sha256),
                required=item.required,
            )
            for item in selected
        )
        manifest = ArtifactManifest(
            artifact_id=spec.id,
            model_id=spec.model_id,
            source=spec.source,
            resolved_revision=resolved_revision,
            format=spec.format,
            quantization=spec.quantization,
            effective_bpw=spec.effective_bpw,
            created_at=created_at,
            files=locked,
            total_logical_bytes=sum(item.size_bytes for item in locked),
            tree_sha256=self._tree_digest(locked),
        )
        return seal_manifest(manifest)

    @staticmethod
    def _tree_digest(files: Sequence[LockedFile]) -> str:
        payload = [
            {
                "logical_path": item.logical_path,
                "required": item.required,
                "role": item.role.value,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in sorted(files, key=lambda value: value.logical_path)
        ]
        return canonical_sha256(payload)

    def _install_blob(
        self, source: Path, digest: str, expected_size: int
    ) -> None:
        destination = self.blob_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._verify_blob(destination, digest, expected_size)
            return

        if self.blob_install_mode == "hardlink":
            try:
                os.link(source, destination, follow_symlinks=True)
            except FileExistsError:
                self._verify_blob(destination, digest, expected_size)
                return
            except OSError:
                pass
            else:
                # chmod applies to the shared inode.  That coupling is why this
                # behavior is opt-in rather than the default for HF snapshots.
                os.chmod(destination, _IMMUTABLE_MODE)
                self._verify_blob(destination, digest, expected_size)
                return

        temporary = destination.parent / f".{digest}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            self._sync_file(temporary)
            os.chmod(temporary, _IMMUTABLE_MODE)
            if sha256_file(temporary) != digest:
                raise IntegrityError(f"source changed while copying blob {digest}")
            if destination.exists():
                self._verify_blob(destination, digest, expected_size)
            else:
                os.replace(temporary, destination)
                self._sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

        self._verify_blob(destination, digest, expected_size)

    def _verify_blob(self, path: Path, digest: str, expected_size: int) -> None:
        self._assert_safe_managed_path(path, self.paths.blob_root)
        if path.is_symlink():
            raise IntegrityError(f"CAS blob must not be a symbolic link: {path}")
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise IntegrityError(f"required CAS blob is missing: {digest}") from exc
        actual_size = metadata.st_size
        if actual_size != expected_size:
            raise IntegrityError(
                f"CAS blob {digest} is {actual_size} bytes; expected {expected_size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != digest:
            raise IntegrityError(
                f"CAS blob corruption at {path}: expected {digest}, "
                f"computed {actual_digest}"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise IntegrityError(
                f"CAS blob {digest} is writable and therefore not immutable: {path}"
            )

    def _write_manifest(
        self, manifest: ArtifactManifest, destination: Path
    ) -> bool:
        self._assert_safe_managed_path(destination, self.paths.manifest_root)
        if destination.is_symlink():
            raise StoragePolicyError(
                f"manifest path must not be a symbolic link: {destination}"
            )
        encoded = canonical_json_bytes(manifest.model_dump(mode="json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._assert_safe_managed_path(destination, self.paths.manifest_root)
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise IntegrityError(
                    f"immutable manifest path contains different bytes: {destination}"
                )
            return False

        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, _IMMUTABLE_MODE)
            if destination.exists():
                if destination.read_bytes() != encoded:
                    raise IntegrityError(
                        f"immutable manifest path raced with different bytes: "
                        f"{destination}"
                    )
                return False
            os.replace(temporary, destination)
            self._sync_directory(destination.parent)
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def _create_view(self, manifest: ArtifactManifest, destination: Path) -> bool:
        self._assert_safe_managed_path(destination, self.paths.view_root)
        if destination.is_symlink():
            raise StoragePolicyError(
                f"artifact view must not be a symbolic link: {destination}"
            )
        if destination.exists():
            if not self._view_matches(
                manifest,
                destination,
                require_immutable=False,
            ):
                raise IntegrityError(
                    f"existing immutable view differs from manifest: {destination}"
                )
            self._freeze_view(destination)
            if not self._view_matches(manifest, destination):
                raise IntegrityError(
                    f"existing view could not be made immutable: {destination}"
                )
            return False

        temporary = self.paths.view_root / (
            f".{manifest.artifact_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            for item in manifest.files:
                target = temporary / PurePosixPath(item.logical_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                blob = self.blob_path(item.sha256)
                relative = os.path.relpath(blob, target.parent)
                try:
                    target.symlink_to(relative)
                except OSError:
                    try:
                        os.link(blob, target)
                    except OSError:
                        shutil.copyfile(blob, target)
                        os.chmod(target, _IMMUTABLE_MODE)
            self._set_view_directory_mode(temporary, _IMMUTABLE_DIRECTORY_MODE)
            self._sync_view_directories(temporary)
            try:
                temporary.rename(destination)
            except FileExistsError:
                if not self._view_matches(manifest, destination):
                    raise IntegrityError(
                        f"view publication raced with different content: {destination}"
                    )
                return False
            self._sync_directory(self.paths.view_root)
            return True
        finally:
            if temporary.exists():
                # A failed publication/race still needs to be recoverable.
                self._set_view_directory_mode(temporary, 0o750)
                shutil.rmtree(temporary)

    @staticmethod
    def _set_view_directory_mode(root: Path, mode: int) -> None:
        directories = [
            path
            for path in root.rglob("*")
            if not path.is_symlink() and path.is_dir()
        ]
        # Freeze children before parents; thaw parents before children.
        directories.sort(
            key=lambda item: len(item.relative_to(root).parts),
            reverse=mode == _IMMUTABLE_DIRECTORY_MODE,
        )
        if mode != _IMMUTABLE_DIRECTORY_MODE:
            os.chmod(root, mode)
        for directory in directories:
            os.chmod(directory, mode)
        if mode == _IMMUTABLE_DIRECTORY_MODE:
            os.chmod(root, mode)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise StoragePolicyError(f"could not durably sync directory {path}: {exc}") from exc

    @classmethod
    def _sync_view_directories(cls, root: Path) -> None:
        directories = [root]
        directories.extend(
            path
            for path in root.rglob("*")
            if not path.is_symlink() and path.is_dir()
        )
        directories.sort(
            key=lambda item: len(item.relative_to(root).parts), reverse=True
        )
        for directory in directories:
            cls._sync_directory(directory)

    @classmethod
    def _freeze_view(cls, root: Path) -> None:
        for path in root.rglob("*"):
            if path.is_symlink() or path.is_dir():
                continue
            os.chmod(path, _IMMUTABLE_MODE)
        cls._set_view_directory_mode(root, _IMMUTABLE_DIRECTORY_MODE)

    def repair_view_permissions(
        self,
        artifact: str | Path | ArtifactManifest | Mapping[str, Any],
    ) -> VerificationReport:
        """Freeze a byte-identical legacy view, then perform strict verification."""

        manifest = self._resolve_manifest(artifact)
        # Establish manifest and CAS integrity independently of the view first.
        self.verify(manifest, verify_view=False)
        view = self.view_path(manifest.artifact_id)
        if not self._view_matches(manifest, view, require_immutable=False):
            raise IntegrityError(
                f"refusing to repair a view whose content differs: {view}"
            )
        self._freeze_view(view)
        self._sync_view_directories(view)
        self._sync_directory(self.paths.view_root)
        return self.verify(manifest, verify_view=True)

    def _view_matches(
        self,
        manifest: ArtifactManifest,
        view: Path,
        *,
        require_immutable: bool = True,
    ) -> bool:
        self._assert_safe_managed_path(view, self.paths.view_root)
        if view.is_symlink() or not view.is_dir():
            return False
        directories = [view]
        directories.extend(
            path
            for path in view.rglob("*")
            if not path.is_symlink() and path.is_dir()
        )
        for directory in directories:
            try:
                metadata = directory.lstat()
            except OSError:
                return False
            if not stat.S_ISDIR(metadata.st_mode) or (
                require_immutable and stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                return False
        expected = {item.logical_path: item for item in manifest.files}
        actual = {
            path.relative_to(view).as_posix()
            for path in view.rglob("*")
            # Check the link itself before any operation that would follow it.
            if path.is_symlink() or path.is_file()
        }
        if actual != set(expected):
            return False
        for logical, item in expected.items():
            path = view / PurePosixPath(logical)
            blob = self.blob_path(item.sha256)
            try:
                # The final member may intentionally be a CAS symlink; validate
                # lexical containment and all parent directories before following
                # it only after checking its link text below.
                self._assert_safe_managed_path(path.parent, view)
            except StoragePolicyError:
                return False
            if path.is_symlink():
                try:
                    link = Path(os.readlink(path))
                except OSError:
                    return False
                target = link if link.is_absolute() else path.parent / link
                if Path(os.path.abspath(target)) != Path(os.path.abspath(blob)):
                    return False
                continue
            try:
                metadata = path.lstat()
            except OSError:
                return False
            if not stat.S_ISREG(metadata.st_mode) or (
                require_immutable and stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                return False
            try:
                if os.path.samefile(path, blob):
                    continue
            except (FileNotFoundError, OSError):
                return False
            if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
                return False
        return True

    def _resolve_manifest(
        self, artifact: str | Path | ArtifactManifest | Mapping[str, Any]
    ) -> ArtifactManifest:
        if isinstance(artifact, ArtifactManifest):
            return artifact
        if isinstance(artifact, Mapping):
            return ArtifactManifest.model_validate(artifact)
        if isinstance(artifact, Path):
            path = artifact
        else:
            # Strings are artifact IDs.  Treating a same-named cwd file as a
            # manifest lets an unrelated valid manifest bypass verification of
            # the requested registered artifact.  File verification is explicit
            # through the Path type only.
            path = None
        if path is not None:
            try:
                return ArtifactManifest.model_validate_json(path.read_text("utf-8"))
            except (OSError, ValueError) as exc:
                raise IntegrityError(f"invalid artifact manifest {path}: {exc}") from exc
        return self.registry.get_artifact(str(artifact))

    def _verify_manifest_file(self, manifest: ArtifactManifest) -> None:
        registered = self.registry.find_artifact(manifest.artifact_id)
        path: Path | None = None
        if registered is not None:
            if registered.manifest_sha256 != manifest.manifest_sha256:
                raise IntegrityError(
                    f"registry manifest differs for {manifest.artifact_id!r}"
                )
            path = self.registry.get_manifest_path(manifest.artifact_id)
        deterministic = self.manifest_path(manifest)
        if path is None and deterministic.exists():
            path = deterministic
        if path is None:
            return

        self._assert_safe_managed_path(path, self.paths.manifest_root)
        if path.is_symlink():
            raise IntegrityError(
                f"persisted manifest must not be a symbolic link: {path}"
            )

        expected = canonical_json_bytes(manifest.model_dump(mode="json"))
        try:
            metadata = path.stat()
            actual = path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"persisted manifest is unavailable: {path}") from exc
        if actual != expected:
            raise IntegrityError(f"persisted manifest is corrupt or noncanonical: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise IntegrityError(f"persisted manifest is writable: {path}")

    def _live_manifests(
        self, keep_artifact_ids: Iterable[str] | None
    ) -> tuple[ArtifactManifest, ...]:
        if keep_artifact_ids is not None:
            identifiers = tuple(dict.fromkeys(keep_artifact_ids))
            return tuple(self.registry.get_artifact(item) for item in identifiers)

        manifests: dict[str, ArtifactManifest] = {
            manifest.manifest_sha256 or "": manifest
            for manifest in self.registry.list_artifacts()
        }
        for path in sorted(self.paths.manifest_root.rglob("*.json")):
            if path.is_symlink():
                raise IntegrityError(
                    f"refusing GC because manifest is a symbolic link: {path}"
                )
            self._assert_safe_managed_path(path, self.paths.manifest_root)
            try:
                parsed = ArtifactManifest.model_validate_json(path.read_text("utf-8"))
                sealed = seal_manifest(parsed)
            except (OSError, ValueError, IntegrityError) as exc:
                raise IntegrityError(
                    f"refusing GC because manifest cannot be validated: {path}: {exc}"
                ) from exc
            manifests[sealed.manifest_sha256 or ""] = sealed
        return tuple(manifests.values())

    def _ensure_free_space(self, target: Path, planned_bytes: int) -> None:
        if planned_bytes < 0:
            raise ValueError("planned_bytes cannot be negative")
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        required = planned_bytes + self.free_reserve_bytes
        if free < required:
            raise StoragePolicyError(
                f"storage policy requires {required} free bytes before the "
                f"operation ({planned_bytes} bytes plus {self.free_reserve_bytes} "
                f"reserve), but {target} has {free}"
            )

    @staticmethod
    def _sync_file(path: Path) -> None:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())

    @contextmanager
    def _storage_lock(self) -> Iterator[None]:
        with self._thread_lock:
            self._assert_managed_layout()
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._assert_safe_managed_path(
                self._lock_path, self.paths.data_root / "state"
            )
            if self._lock_path.is_symlink():
                raise StoragePolicyError(
                    f"storage lock must not be a symbolic link: {self._lock_path}"
                )
            with self._lock_path.open("a+b") as lock_file:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows
                    yield
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _assert_managed_layout(self) -> None:
        """Reject symbolic links in directories managed by this store."""

        data_root = Path(os.path.abspath(self.paths.data_root.expanduser()))
        if data_root.is_symlink():
            raise StoragePolicyError(
                f"data root must not be a symbolic link: {data_root}"
            )
        for relative in DATA_DIRECTORIES:
            self._assert_safe_managed_path(data_root / relative, data_root)
        self._assert_safe_managed_path(self.paths.registry_path, data_root)
        if self.paths.registry_path.is_symlink():
            raise StoragePolicyError(
                f"registry database must not be a symbolic link: "
                f"{self.paths.registry_path}"
            )

    @staticmethod
    def _assert_safe_managed_path(path: Path, root: Path) -> None:
        """Require *path* to stay below *root* without existing symlink hops."""

        # abspath collapses ``.`` and ``..`` lexically but, unlike resolve(),
        # does not dereference any symlink before we have inspected it.
        managed_root = Path(os.path.abspath(root.expanduser()))
        candidate = Path(os.path.abspath(path.expanduser()))
        try:
            relative = candidate.relative_to(managed_root)
        except ValueError as exc:
            raise StoragePolicyError(
                f"managed storage path escapes {managed_root}: {candidate}"
            ) from exc

        current = managed_root
        if current.is_symlink():
            raise StoragePolicyError(
                f"managed storage path contains a symbolic link: {current}"
            )
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise StoragePolicyError(
                    f"managed storage path contains a symbolic link: {current}"
                )


# Descriptive aliases used by scripts and older prototypes.
CASStore = ArtifactStore
StorageManager = ArtifactStore


def _validate_pattern(pattern: str) -> None:
    if not pattern or "\x00" in pattern:
        raise CatalogError("artifact selector pattern must be non-empty and non-NUL")
    pure = PurePosixPath(pattern)
    if pure.is_absolute() or ".." in pure.parts:
        raise CatalogError(f"unsafe artifact selector pattern {pattern!r}")


def artifact_path_matches(logical_path: str, pattern: str) -> bool:
    """Match an artifact path from its root, with segment-aware ``**`` globs.

    ``PurePath.match`` right-anchors patterns such as ``model.gguf`` and would
    therefore also accept ``sub/model.gguf``.  Artifact selectors are rooted:
    a pattern without a directory component must only match a root-level file.
    """

    _validate_pattern(pattern)
    logical = PurePosixPath(logical_path)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        return False
    path_parts = logical.parts
    pattern_parts = PurePosixPath(pattern).parts
    memo: dict[tuple[int, int], bool] = {}

    def matches(path_index: int, pattern_index: int) -> bool:
        key = (path_index, pattern_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
        elif pattern_parts[pattern_index] == "**":
            result = matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts)
                and matches(path_index + 1, pattern_index)
            )
        else:
            result = (
                path_index < len(path_parts)
                and fnmatch.fnmatchcase(
                    path_parts[path_index], pattern_parts[pattern_index]
                )
                and matches(path_index + 1, pattern_index + 1)
            )
        memo[key] = result
        return result

    return matches(0, 0)


def _remove_empty_parent(path: Path, stop: Path) -> None:
    if path == stop:
        return
    try:
        path.rmdir()
    except OSError:
        pass
