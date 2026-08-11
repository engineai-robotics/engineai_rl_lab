import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import yaml

from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms


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


def _load_motion_list_from_yaml(
    yaml_path: Path,
) -> tuple[list[tuple[str, float, tuple[float, float, float] | None]], list[int] | None]:
    """Parse YAML motion paths, weights, conditional commands, and body metadata."""
    if not yaml_path.is_file():
        raise AssertionError(f"Invalid YAML path: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    motions = data.get("motions", [])
    base_dir = yaml_path.parent
    motion_specs: list[tuple[str, float, tuple[float, float, float] | None]] = []
    foot_body_indices = data.get("foot_body_indices")
    if foot_body_indices is not None:
        if not isinstance(foot_body_indices, list) or len(foot_body_indices) != 2:
            raise ValueError("foot_body_indices must contain exactly two body indices.")
        foot_body_indices = [int(index) for index in foot_body_indices]

    for entry in motions:
        weight = float(entry.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Motion weight must be finite and positive, got {weight}: {entry}")
        command = entry.get("command")
        if command is not None:
            if not isinstance(command, list) or len(command) != 3:
                raise ValueError(f"Motion command must be [vx, vy, wz], got: {command}")
            command = tuple(float(value) for value in command)
            if not all(math.isfinite(value) for value in command):
                raise ValueError(f"Motion command must be finite, got: {command}")
        if "file" in entry:
            file_path = (base_dir / entry["file"]).resolve()
            if not file_path.is_file():
                raise AssertionError(f"Motion file not found: {file_path}")
            motion_specs.append((str(file_path), weight, command))
        elif "folder" in entry:
            folder_path = (base_dir / entry["folder"]).resolve()
            if not folder_path.is_dir():
                raise AssertionError(f"Motion folder not found: {folder_path}")
            npz_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".npz")
            if not npz_files:
                raise AssertionError(f"No .npz files found in folder: {folder_path}")
            motion_specs.extend((str(file_path), weight, command) for file_path in npz_files)
        else:
            raise ValueError(f"Each motion entry must contain 'file' or 'folder': {entry}")
    return motion_specs, foot_body_indices


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
    waist_indices = [
        idx for idx, name in enumerate(filtered_names) if "WAIST_YAW" in name or "TORSO_YAW" in name
    ]
    if waist_indices:
        joint_pos = joint_pos.copy()
        joint_vel = joint_vel.copy()
        joint_pos[:, waist_indices] = 0.0
        joint_vel[:, waist_indices] = 0.0
    return joint_pos, joint_vel, filtered_names


def _canonicalize_joint_names(joint_names: list[str] | None) -> list[str] | None:
    """Normalize known cross-robot joint aliases for consistency checks."""
    if joint_names is None:
        return None
    return [name.replace("TORSO_YAW", "WAIST_YAW") for name in joint_names]


def _validate_motion_arrays(
    file_path: str,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
) -> None:
    """Validate one motion clip before adding it to the expert dataset."""
    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"Invalid joint arrays in {file_path}: joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}"
        )

    num_frames = joint_pos.shape[0]
    expected_shapes = {
        "body_pos_w": (num_frames, None, 3),
        "body_quat_w": (num_frames, None, 4),
        "body_lin_vel_w": (num_frames, None, 3),
        "body_ang_vel_w": (num_frames, None, 3),
    }
    body_arrays = {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    num_bodies = body_pos_w.shape[1] if body_pos_w.ndim == 3 else None
    for name, array in body_arrays.items():
        expected = expected_shapes[name]
        shape_is_invalid = (
            array.ndim != 3
            or array.shape[0] != expected[0]
            or array.shape[1] != num_bodies
            or array.shape[2] != expected[2]
        )
        if shape_is_invalid:
            raise ValueError(f"Invalid {name} shape in {file_path}: got {array.shape}")

    for name, array in {"joint_pos": joint_pos, "joint_vel": joint_vel, **body_arrays}.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values found in {name} of {file_path}")


class AMPDataLoader:
    def __init__(
        self,
        motion_file: str | list[str],
        device: str = "cpu",
        history_length: int = 5,
        include_joint_vel: bool = False,
        include_foot_features: bool = False,
        condition_dim: int = 0,
    ):
        assert history_length >= 1, "history_length must be positive"
        if condition_dim not in (0, 3):
            raise ValueError(f"condition_dim must be 0 or 3, got {condition_dim}.")

        motion_specs: list[tuple[str, float, tuple[float, float, float] | None]] = []
        foot_body_indices: list[int] | None = None
        if isinstance(motion_file, str):
            motion_path = _resolve_repo_path(motion_file)
            if motion_path.suffix == ".yaml":
                yaml_specs, foot_body_indices = _load_motion_list_from_yaml(motion_path)
                motion_specs.extend(yaml_specs)
            elif motion_path.is_file() and motion_path.suffix == ".npz":
                motion_specs.append((str(motion_path), 1.0, None))
            elif motion_path.is_dir():
                for file_name in sorted(os.listdir(motion_path)):
                    full_path = motion_path / file_name
                    if full_path.suffix == ".npz" and full_path.is_file():
                        motion_specs.append((str(full_path), 1.0, None))
            else:
                raise AssertionError(f"Invalid motion source: {motion_file}")
        elif isinstance(motion_file, list):
            for file_name in motion_file:
                motion_path = _resolve_repo_path(file_name)
                if motion_path.suffix == ".yaml":
                    yaml_specs, yaml_foot_body_indices = _load_motion_list_from_yaml(motion_path)
                    motion_specs.extend(yaml_specs)
                    if foot_body_indices is None:
                        foot_body_indices = yaml_foot_body_indices
                    elif yaml_foot_body_indices != foot_body_indices:
                        raise ValueError("All motion YAML files must use the same foot_body_indices.")
                elif motion_path.is_file() and motion_path.suffix == ".npz":
                    motion_specs.append((str(motion_path), 1.0, None))

        assert len(motion_specs) > 0, f"No valid motion data found in: {motion_file}"
        if condition_dim > 0 and any(command is None for _, _, command in motion_specs):
            raise ValueError("Every motion requires a [vx, vy, wz] command label for conditional AMP.")
        if include_foot_features and foot_body_indices is None:
            raise ValueError("Conditional foot features require foot_body_indices in the motion YAML.")
        print("\n=========== AMP Motion File List ===========")
        for idx, (path, weight, command) in enumerate(motion_specs):
            print(f"{idx + 1:2d}. weight={weight:g} command={command}  {path}")
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
        motion_commands: list[tuple[float, float, float]] = []
        joint_names: list[str] | None = None
        canonical_joint_names: list[str] | None = None
        expected_fps: float | None = None
        expected_num_bodies: int | None = None
        loaded_motion_paths: list[str] = []

        for file_path, motion_weight, motion_command in motion_specs:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    required_keys = {
                        "fps",
                        "joint_pos",
                        "joint_vel",
                        "body_pos_w",
                        "body_quat_w",
                        "body_lin_vel_w",
                        "body_ang_vel_w",
                        "joint_names",
                    }
                    missing_keys = required_keys.difference(data.files)
                    if missing_keys:
                        raise KeyError(f"Missing required arrays: {sorted(missing_keys)}")

                    joint_pos, joint_vel, file_joint_names = _load_joint_arrays(data)
                    file_canonical_joint_names = _canonicalize_joint_names(file_joint_names)
                    if canonical_joint_names is None:
                        joint_names = file_joint_names
                        canonical_joint_names = file_canonical_joint_names
                    elif canonical_joint_names != file_canonical_joint_names:
                        raise ValueError(
                            f"Joint name order mismatch: expected {canonical_joint_names}, "
                            f"got {file_canonical_joint_names}"
                        )

                    fps = float(data["fps"])
                    if not math.isfinite(fps) or fps <= 0.0:
                        raise ValueError(f"Invalid fps: {fps}")
                    if expected_fps is None:
                        expected_fps = fps
                    elif not math.isclose(fps, expected_fps, rel_tol=1.0e-6):
                        raise ValueError(f"FPS mismatch: expected {expected_fps}, got {fps}")

                    body_pos_w = data["body_pos_w"]
                    body_quat_w = data["body_quat_w"]
                    body_lin_vel_w = data["body_lin_vel_w"]
                    body_ang_vel_w = data["body_ang_vel_w"]
                    _validate_motion_arrays(
                        file_path,
                        joint_pos,
                        joint_vel,
                        body_pos_w,
                        body_quat_w,
                        body_lin_vel_w,
                        body_ang_vel_w,
                    )
                    num_bodies = body_pos_w.shape[1]
                    if expected_num_bodies is None:
                        expected_num_bodies = num_bodies
                    elif num_bodies != expected_num_bodies:
                        raise ValueError(f"Body count mismatch: expected {expected_num_bodies}, got {num_bodies}")

                    fps_list.append(fps)
                    joint_pos_list.append(torch.tensor(joint_pos, dtype=torch.float32, device=device))
                    joint_vel_list.append(torch.tensor(joint_vel, dtype=torch.float32, device=device))
                    body_pos_w_list.append(torch.tensor(body_pos_w, dtype=torch.float32, device=device))
                    body_quat_w_list.append(torch.tensor(body_quat_w, dtype=torch.float32, device=device))
                    body_lin_vel_w_list.append(torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device))
                    body_ang_vel_w_list.append(torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device))
                motion_weights.append(motion_weight)
                motion_lengths.append(joint_pos.shape[0])
                motion_commands.append(motion_command or (0.0, 0.0, 0.0))
                loaded_motion_paths.append(file_path)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to load AMP motion file '{file_path}': {exc}") from exc

        if not joint_pos_list:
            raise RuntimeError("Failed to load any AMP motion data")
        print(
            f"Successfully loaded all {len(loaded_motion_paths)} AMP motion files "
            f"({sum(motion_lengths)} frames at {expected_fps:g} FPS)."
        )
        self.fps = torch.tensor(fps_list, dtype=torch.float32, device=device)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        self.joint_names = joint_names
        self.loaded_motion_paths = loaded_motion_paths
        self.time_step_total = self.joint_pos.shape[0]
        self.motion_weights = torch.tensor(motion_weights, dtype=torch.float32, device=device)
        self.motion_lengths = torch.tensor(motion_lengths, dtype=torch.long, device=device)
        self.motion_commands = torch.tensor(motion_commands, dtype=torch.float32, device=device)
        motion_starts = torch.cumsum(self.motion_lengths, dim=0) - self.motion_lengths
        self.motion_starts = motion_starts
        self.frame_motion_start = torch.repeat_interleave(motion_starts, self.motion_lengths)
        self.frame_motion_id = torch.repeat_interleave(
            torch.arange(len(motion_lengths), device=device),
            self.motion_lengths,
        )
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
        self.include_joint_vel = include_joint_vel
        self.include_foot_features = include_foot_features
        self.condition_dim = condition_dim
        self.foot_body_indices = (
            torch.tensor(foot_body_indices, dtype=torch.long, device=device)
            if foot_body_indices is not None
            else None
        )
        if self.foot_body_indices is not None:
            if torch.any(self.foot_body_indices < 0) or torch.any(self.foot_body_indices >= self.num_bodies):
                raise ValueError(
                    f"foot_body_indices {foot_body_indices} are invalid for {self.num_bodies} bodies."
                )

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

    def _gather_frame_features(self, idxs: torch.Tensor) -> list[torch.Tensor]:
        """Collect one physical AMP frame in the same order as the environment."""
        features = [self.joint_pos[idxs] * 9]
        if self.include_joint_vel:
            features.append(self.joint_vel[idxs])
        features.extend(
            [
                self.base_lin_vel_b[idxs] * 7,
                self.base_ang_vel_b[idxs],
                self.projected_gravity_b[idxs],
            ]
        )
        if self.include_foot_features:
            features.extend(
                [
                    self.body_pos_b[idxs][:, self.foot_body_indices].flatten(start_dim=1),
                    self.body_lin_vel_b[idxs][:, self.foot_body_indices].flatten(start_dim=1),
                ]
            )
        return features

    def _gather_history(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Gather an oldest-to-newest history without crossing clip boundaries."""
        history_offsets = torch.arange(
            -(self.history_length - 1),
            1,
            device=self.joint_pos.device,
        )
        motion_starts = self.frame_motion_start[batch_indices]
        frame_features = []
        for offset in history_offsets:
            idxs = torch.maximum(batch_indices + offset, motion_starts)
            frame_features.extend(self._gather_frame_features(idxs))
        return torch.cat(frame_features, dim=-1)

    def condition_support_mask(self, commands: torch.Tensor) -> torch.Tensor:
        """Exclude standing commands until the dataset contains a standing expert clip."""
        if self.condition_dim == 0:
            return torch.ones(commands.shape[0], dtype=torch.bool, device=commands.device)
        return torch.linalg.vector_norm(commands, dim=1) > 0.1

    def sample_conditioned(
        self,
        commands: torch.Tensor,
        max_samples: int | None = None,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """Sample expert histories from clips whose YAML labels match policy commands."""
        if self.condition_dim == 0:
            raise RuntimeError("sample_conditioned requires condition_dim > 0.")
        if commands.ndim != 2 or commands.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected commands with shape [N, {self.condition_dim}], got {tuple(commands.shape)}."
            )
        if commands.shape[0] == 0:
            raise ValueError("Cannot sample conditional AMP data for an empty command batch.")

        if max_samples is not None and commands.shape[0] > max_samples:
            selection = torch.randperm(commands.shape[0], device=commands.device)[:max_samples]
            commands = commands[selection]

        squared_distance = torch.sum(
            torch.square(commands.unsqueeze(1) - self.motion_commands.unsqueeze(0)),
            dim=-1,
        )
        logits = -squared_distance / max(temperature, 1.0e-6)
        logits = logits + torch.log(self.motion_weights).unsqueeze(0)
        motion_probabilities = torch.softmax(logits, dim=1)
        motion_ids = torch.multinomial(motion_probabilities, num_samples=1).squeeze(1)

        motion_lengths = self.motion_lengths[motion_ids]
        frame_offsets = torch.floor(
            torch.rand(commands.shape[0], device=commands.device) * motion_lengths
        ).to(torch.long)
        batch_indices = self.motion_starts[motion_ids] + frame_offsets
        return torch.cat([self._gather_history(batch_indices), commands], dim=-1)

    def mini_batch_generator(self, num_mini_batches, num_epoches) -> Iterator[torch.Tensor]:
        """Generate mini-batches of motion data."""
        num_samples = self.joint_pos.shape[0]
        batch_size = math.ceil(num_samples / num_mini_batches)

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
                batch = self._gather_history(batch_indices)
                if self.condition_dim > 0:
                    commands = self.motion_commands[self.frame_motion_id[batch_indices]]
                    batch = torch.cat([batch, commands], dim=-1)
                yield batch
