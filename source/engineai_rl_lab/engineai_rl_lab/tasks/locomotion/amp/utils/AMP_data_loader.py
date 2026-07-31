import math
import numpy as np
import os
import torch
import yaml
from collections.abc import Iterator
from pathlib import Path

from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_inv


def _find_repo_root(start: Path) -> Path:
    """Find the workspace root so dataset paths can stay repo-relative."""
    for path in (start, *start.parents):
        if (path / "datasets").is_dir() and (path / "source").is_dir():
            return path
    return Path.cwd()


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _resolve_repo_path(path_like: str) -> Path:
    """Resolve path relative to the repository root when not absolute."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _load_motion_list_from_yaml(yaml_path: Path) -> list[tuple[str, float]]:
    """Parse YAML motion config into absolute file paths and sampling weights."""
    if not yaml_path.is_file():
        raise AssertionError(f"Invalid YAML path: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    motions = data.get("motions", [])
    base_dir = yaml_path.parent
    motions_with_weights: list[tuple[str, float]] = []

    for entry in motions:
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Motion weight must be finite and positive, got {weight}: {entry}")
        if "file" in entry:
            file_path = (base_dir / entry["file"]).resolve()
            if not file_path.is_file():
                raise AssertionError(f"Motion file not found: {file_path}")
            motions_with_weights.append((str(file_path), weight))
        elif "folder" in entry:
            folder_path = (base_dir / entry["folder"]).resolve()
            if not folder_path.is_dir():
                raise AssertionError(f"Motion folder not found: {folder_path}")
            npz_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".npz")
            if not npz_files:
                raise AssertionError(f"No .npz files found in folder: {folder_path}")
            motions_with_weights.extend((str(file_path), weight) for file_path in npz_files)
        else:
            raise ValueError(f"Each motion entry must contain 'file' or 'folder': {entry}")
    return motions_with_weights


def _load_joint_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Load joint state arrays, dropping head joints when joint names are available."""
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    if "joint_names" not in data:
        return joint_pos, joint_vel, None

    joint_names = [str(name) for name in data["joint_names"].tolist()]
    keep_indices = [idx for idx, name in enumerate(joint_names) if "HEAD" not in name]
    filtered_names = [joint_names[idx] for idx in keep_indices]
    if len(keep_indices) != len(joint_names):
        joint_pos = joint_pos[:, keep_indices]
        joint_vel = joint_vel[:, keep_indices]
    waist_indices = [idx for idx, name in enumerate(filtered_names) if "WAIST_YAW" in name]
    if waist_indices:
        joint_pos = joint_pos.copy()
        joint_vel = joint_vel.copy()
        joint_pos[:, waist_indices] = 0.0
        joint_vel[:, waist_indices] = 0.0
    return joint_pos, joint_vel, filtered_names


class AMPDataLoader:
    def __init__(
        self,
        motion_file: str | list[str],
        device: str = "cpu",
        history_length: int = 5,
    ):
        assert history_length >= 1, "history_length must be positive"

        motion_specs: list[tuple[str, float]] = []
        if isinstance(motion_file, str):
            motion_path = _resolve_repo_path(motion_file)
            if motion_path.suffix == ".yaml":
                motion_specs.extend(_load_motion_list_from_yaml(motion_path))
            elif motion_path.is_file() and motion_path.suffix == ".npz":
                motion_specs.append((str(motion_path), 1.0))
            elif motion_path.is_dir():
                for file_name in sorted(os.listdir(motion_path)):
                    full_path = motion_path / file_name
                    if full_path.suffix == ".npz" and full_path.is_file():
                        motion_specs.append((str(full_path), 1.0))
            else:
                raise AssertionError(f"Invalid motion source: {motion_file}")
        elif isinstance(motion_file, list):
            for file_name in motion_file:
                motion_path = _resolve_repo_path(file_name)
                if motion_path.suffix == ".yaml":
                    motion_specs.extend(_load_motion_list_from_yaml(motion_path))
                elif motion_path.is_file() and motion_path.suffix == ".npz":
                    motion_specs.append((str(motion_path), 1.0))

        assert len(motion_specs) > 0, f"No valid motion data found in: {motion_file}"
        print("\n=========== AMP Motion File List ===========")
        for idx, (path, weight) in enumerate(motion_specs):
            print(f"{idx + 1:2d}. weight={weight:g}  {path}")
        print(f"=========== Total: {len(motion_specs)} files ===========\n")

        fps_list: list[float] = []
        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        motion_weights: list[float] = []
        motion_lengths: list[int] = []
        joint_names: list[str] | None = None

        for file_path, motion_weight in motion_specs:
            try:
                data = np.load(file_path, allow_pickle=True)
                joint_pos, joint_vel, file_joint_names = _load_joint_arrays(data)
                if file_joint_names is not None:
                    if joint_names is None:
                        joint_names = file_joint_names
                    elif joint_names != file_joint_names:
                        raise ValueError(f"Joint name order mismatch in motion file: {file_path}")
                fps_list.append(float(data["fps"]))
                joint_pos_list.append(torch.tensor(joint_pos, dtype=torch.float32, device=device))
                joint_vel_list.append(torch.tensor(joint_vel, dtype=torch.float32, device=device))
                body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
                body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
                body_lin_vel_w_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
                body_ang_vel_w_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
                motion_weights.append(motion_weight)
                motion_lengths.append(joint_pos.shape[0])
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Could not load {file_path}: {exc}")

        assert len(joint_pos_list) > 0, "Failed to load any motion data"
        self.fps = torch.tensor(fps_list, dtype=torch.float32, device=device)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        self.joint_names = joint_names
        self.time_step_total = self.joint_pos.shape[0]
        self.motion_weights = torch.tensor(motion_weights, dtype=torch.float32, device=device)
        self.motion_lengths = torch.tensor(motion_lengths, dtype=torch.long, device=device)
        motion_starts = torch.cumsum(self.motion_lengths, dim=0) - self.motion_lengths
        self.frame_motion_start = torch.repeat_interleave(motion_starts, self.motion_lengths)
        # Divide each clip's mass over its frames: the total probability of selecting
        # a clip is proportional to its YAML weight, independent of clip duration.
        per_motion_frame_weight = self.motion_weights / self.motion_lengths.to(torch.float32)
        self.frame_sampling_weights = torch.repeat_interleave(per_motion_frame_weight, self.motion_lengths)
        self.body_pos_b = torch.zeros_like(self.body_pos_w, device=device)
        self.body_quat_b = torch.zeros_like(self.body_quat_w, device=device)
        self.body_lin_vel_b = torch.zeros_like(self.body_lin_vel_w, device=device)
        self.body_ang_vel_b = torch.zeros_like(self.body_ang_vel_w, device=device)
        self.projected_gravity_b = torch.zeros((self.time_step_total, 3), dtype=torch.float32, device=device)
        self.num_bodies = self.body_pos_w.shape[1]
        self.history_length = history_length

        # world-frame gravity direction, rotated into the base frame per timestep below
        gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)

        for i in range(self.time_step_total):            
            body_pos_w_t = self.body_pos_w[i].unsqueeze(0)  # (1, B, 3)
            body_quat_w_t = self.body_quat_w[i].unsqueeze(0)  # (1, B, 4)
            achor_pos_w_t = body_pos_w_t[:, 0:1, :]  # (1, 1, 3) 
            achor_quat_w_t = body_quat_w_t[:, 0:1, :]  # (1, 1, 4)
            pos_b_t, quat_b_t = subtract_frame_transforms(
                achor_pos_w_t.repeat(1, self.num_bodies, 1),
                achor_quat_w_t.repeat(1, self.num_bodies, 1),
                body_pos_w_t,
                body_quat_w_t,
            )
            
            self.body_pos_b[i] = pos_b_t.squeeze(0)
            self.body_quat_b[i] = quat_b_t.squeeze(0)
            # rotate velocities into anchor frame with inverse; expand anchor quat per body for broadcasting
            anchor_quat_broadcast = quat_inv(achor_quat_w_t.repeat(1, self.num_bodies, 1)).reshape(-1, 4)
            body_lin_vel = self.body_lin_vel_w[i].unsqueeze(0).reshape(-1, 3)
            self.body_lin_vel_b[i] = quat_apply(anchor_quat_broadcast, body_lin_vel).reshape(self.num_bodies, 3)
            body_ang_vel = self.body_ang_vel_w[i].unsqueeze(0).reshape(-1, 3)
            self.body_ang_vel_b[i] = quat_apply(anchor_quat_broadcast, body_ang_vel).reshape(self.num_bodies, 3)
            # gravity in the base (anchor) frame
            base_quat_inv = quat_inv(achor_quat_w_t.reshape(-1, 4))  # (1, 4)
            self.projected_gravity_b[i] = quat_apply(base_quat_inv, gravity_vec_w).reshape(3)
        self.base_lin_vel_b = self.body_lin_vel_b[:, 0, :]
        self.base_ang_vel_b = self.body_ang_vel_b[:, 0, :]

    # Ugly mini batch generator; the observation acquirement need to be re-designed
    def mini_batch_generator(self, num_mini_batches, num_epoches) -> Iterator[torch.Tensor]:
        """Generate mini-batches of motion data."""
        num_samples = self.joint_pos.shape[0]
        batch_size = math.ceil(num_samples / num_mini_batches)
        # history is ordered old -> new; offsets are negative to zero so the last frame is "current"
        history_offsets = torch.arange(-(self.history_length - 1), 1, device=self.joint_pos.device)

        def gather_frame_features(idxs: torch.Tensor) -> list[torch.Tensor]:
            """Collect per-frame features for the given indices."""
            pos = self.joint_pos[idxs] * 9
            base_lin = self.base_lin_vel_b[idxs] * 7
            base_ang = self.base_ang_vel_b[idxs]
            projected_gravity = self.projected_gravity_b[idxs]
            # order must match dummy_history_term in the env configs
            return [pos, base_lin, base_ang, projected_gravity]

        for _ in range(num_epoches):
            # Sample with replacement so YAML weights control each clip's expected
            # contribution while preserving exactly num_samples reference samples.
            indices = torch.multinomial(
                self.frame_sampling_weights,
                num_samples=num_samples,
                replacement=True,
            )
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                motion_starts = self.frame_motion_start[batch_indices]

                frame_features = []
                # history direction: old -> new (last frame aligns with batch_indices)
                for offset in history_offsets:
                    # Clamp at this clip's first frame, never into the previous motion.
                    idxs = torch.maximum(batch_indices + offset, motion_starts)
                    frame_features.extend(gather_frame_features(idxs))

                yield torch.cat(frame_features, dim=-1)
