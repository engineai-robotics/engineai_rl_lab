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


def _load_motion_list_from_yaml(yaml_path: Path) -> list[str]:
    """Parse YAML motion config into absolute file paths."""
    if not yaml_path.is_file():
        raise AssertionError(f"Invalid YAML path: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    motions = data.get("motions", [])
    base_dir = yaml_path.parent
    paths: list[str] = []

    for entry in motions:
        if "file" in entry:
            file_path = (base_dir / entry["file"]).resolve()
            if not file_path.is_file():
                raise AssertionError(f"Motion file not found: {file_path}")
            paths.append(str(file_path))
        elif "folder" in entry:
            folder_path = (base_dir / entry["folder"]).resolve()
            if not folder_path.is_dir():
                raise AssertionError(f"Motion folder not found: {folder_path}")
            npz_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".npz")
            if not npz_files:
                raise AssertionError(f"No .npz files found in folder: {folder_path}")
            paths.extend(str(file_path) for file_path in npz_files)
    return paths


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

        file_list: list[str] = []
        if isinstance(motion_file, str):
            motion_path = _resolve_repo_path(motion_file)
            if motion_path.suffix == ".yaml":
                file_list.extend(_load_motion_list_from_yaml(motion_path))
            elif motion_path.is_file() and motion_path.suffix == ".npz":
                file_list.append(str(motion_path))
            elif motion_path.is_dir():
                for file_name in os.listdir(motion_path):
                    full_path = motion_path / file_name
                    if full_path.suffix == ".npz" and full_path.is_file():
                        file_list.append(str(full_path))
            else:
                raise AssertionError(f"Invalid motion source: {motion_file}")
        elif isinstance(motion_file, list):
            for file_name in motion_file:
                motion_path = _resolve_repo_path(file_name)
                if motion_path.suffix == ".yaml":
                    file_list.extend(_load_motion_list_from_yaml(motion_path))
                elif motion_path.is_file() and motion_path.suffix == ".npz":
                    file_list.append(str(motion_path))

        assert len(file_list) > 0, f"No valid motion data found in: {motion_file}"
        print("\n=========== AMP Motion File List ===========")
        for idx, path in enumerate(file_list):
            print(f"{idx + 1:2d}. {path}")
        print(f"=========== Total: {len(file_list)} files ===========\n")

        fps_list: list[float] = []
        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        joint_names: list[str] | None = None

        for file_path in file_list:
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
        self.body_pos_b = torch.zeros_like(self.body_pos_w, device=device)
        self.body_quat_b = torch.zeros_like(self.body_quat_w, device=device)
        self.body_lin_vel_b = torch.zeros_like(self.body_lin_vel_w, device=device)
        self.body_ang_vel_b = torch.zeros_like(self.body_ang_vel_w, device=device)
        self.projected_gravity_b = torch.zeros((self.time_step_total, 3), dtype=torch.float32, device=device)
        self.num_bodies = self.body_pos_w.shape[1]
        self.history_length = history_length

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
        self.base_lin_vel_b = self.body_lin_vel_b[:, 0, :]

    # Ugly mini batch generator; the observation acquirement need to be re-designed
    def mini_batch_generator(self, num_mini_batches, num_epoches) -> Iterator[torch.Tensor]:
        """Generate mini-batches of motion data."""
        num_samples = self.joint_pos.shape[0]
        batch_size = math.ceil(num_samples / num_mini_batches)
        indices = torch.randperm(num_samples, device=self.joint_pos.device)
        # history is ordered old -> new; offsets are negative to zero so the last frame is "current"
        history_offsets = torch.arange(-(self.history_length - 1), 1, device=self.joint_pos.device)

        def gather_frame_features(idxs: torch.Tensor) -> list[torch.Tensor]:
            """Collect per-frame features for the given indices."""
            pos = self.joint_pos[idxs] * 9
            base_lin = self.base_lin_vel_b[idxs] * 7
            return [pos, base_lin]

        for _ in range(num_epoches):
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]

                frame_features = []
                # history direction: old -> new (last frame aligns with batch_indices)
                for offset in history_offsets:
                    idxs = torch.clamp(batch_indices + offset, min=0, max=num_samples - 1)
                    frame_features.extend(gather_frame_features(idxs))

                yield torch.cat(frame_features, dim=-1)
