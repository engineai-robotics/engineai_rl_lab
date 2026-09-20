import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import yaml

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_unique,
)


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
) -> tuple[
    list[tuple[str, float, tuple[float, float, float] | None]],
    list[int] | None,
    list[int],
]:
    """Parse compact motion paths, weights, commands, and DOF alignment metadata."""
    if not yaml_path.is_file():
        raise AssertionError(f"Invalid YAML path: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    motions = data.get("motions", [])
    base_dir = yaml_path.parent
    motion_specs: list[tuple[str, float, tuple[float, float, float] | None]] = []
    dof_indices = data.get("dof_indices")
    if dof_indices is not None:
        if not isinstance(dof_indices, list) or not dof_indices:
            raise ValueError("dof_indices must be a non-empty list.")
        dof_indices = [int(index) for index in dof_indices]
        if any(index < 0 for index in dof_indices):
            raise ValueError("dof_indices cannot contain negative indices.")
        if len(set(dof_indices)) != len(dof_indices):
            raise ValueError("dof_indices cannot contain duplicate indices.")

    zero_dof_indices = data.get("zero_dof_indices", [])
    if not isinstance(zero_dof_indices, list):
        raise ValueError("zero_dof_indices must be a list.")
    zero_dof_indices = [int(index) for index in zero_dof_indices]
    output_dof_count = len(dof_indices) if dof_indices is not None else None
    if any(index < 0 or (output_dof_count is not None and index >= output_dof_count) for index in zero_dof_indices):
        raise ValueError("zero_dof_indices must refer to the aligned output DOFs.")

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
    return motion_specs, dof_indices, zero_dof_indices


def _forward_difference(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute forward finite differences, repeating the final velocity."""
    if data.shape[0] < 2:
        raise ValueError("A motion must contain at least two frames.")
    velocity = torch.empty_like(data)
    velocity[:-1] = (data[1:] - data[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def _angular_velocity_world(root_quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute world-frame angular velocity from a wxyz quaternion trajectory."""
    relative_quat = quat_mul(quat_conjugate(root_quat[:-1]), root_quat[1:])
    relative_quat = quat_unique(relative_quat)
    angular_velocity_body = axis_angle_from_quat(relative_quat) / dt
    angular_velocity_world = torch.empty(
        (root_quat.shape[0], 3),
        dtype=root_quat.dtype,
        device=root_quat.device,
    )
    angular_velocity_world[:-1] = quat_apply(root_quat[:-1], angular_velocity_body)
    angular_velocity_world[-1] = angular_velocity_world[-2]
    return angular_velocity_world


def _validate_compact_motion(
    file_path: str,
    fps: float,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
) -> None:
    """Validate a compact GMR-style motion clip."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid fps in {file_path}: {fps}")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Invalid root_pos shape in {file_path}: {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Invalid root_rot shape in {file_path}: {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Invalid dof_pos shape in {file_path}: {dof_pos.shape}")
    num_frames = dof_pos.shape[0]
    if num_frames < 2:
        raise ValueError(f"Motion must contain at least two frames: {file_path}")
    if root_pos.shape[0] != num_frames or root_rot.shape[0] != num_frames:
        raise ValueError(
            f"Frame count mismatch in {file_path}: "
            f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={num_frames}"
        )
    for name, array in {"root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos}.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values found in {name} of {file_path}")
    quat_norm = np.linalg.norm(root_rot, axis=1)
    if np.any(quat_norm < 1.0e-6):
        raise ValueError(f"Zero-length root quaternion found in {file_path}")


class AMPDataLoader:
    def __init__(
        self,
        motion_file: str | list[str],
        device: str = "cpu",
        history_length: int = 5,
        include_joint_vel: bool = False,
        include_base_lin_vel: bool = True,
        include_projected_gravity: bool = True,
        include_foot_features: bool = False,
        flatten_history_dim: bool = True,
        condition_dim: int = 0,
    ):
        assert history_length >= 1, "history_length must be positive"
        if condition_dim not in (0, 3):
            raise ValueError(f"condition_dim must be 0 or 3, got {condition_dim}.")
        if condition_dim > 0 and not flatten_history_dim:
            raise ValueError("Conditional AMP currently requires flatten_history_dim=True.")
        if include_foot_features:
            raise ValueError(
                "Compact AMP motions do not contain rigid-body states, so include_foot_features is unsupported."
            )

        motion_specs: list[tuple[str, float, tuple[float, float, float] | None]] = []
        dof_indices: list[int] | None = None
        zero_dof_indices: list[int] = []
        if isinstance(motion_file, str):
            motion_path = _resolve_repo_path(motion_file)
            if motion_path.suffix == ".yaml":
                yaml_specs, dof_indices, zero_dof_indices = _load_motion_list_from_yaml(motion_path)
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
                    yaml_specs, yaml_dof_indices, yaml_zero_dof_indices = _load_motion_list_from_yaml(motion_path)
                    motion_specs.extend(yaml_specs)
                    if dof_indices is None:
                        dof_indices = yaml_dof_indices
                    elif yaml_dof_indices is not None and yaml_dof_indices != dof_indices:
                        raise ValueError("All motion YAML files must use the same dof_indices.")
                    if not zero_dof_indices:
                        zero_dof_indices = yaml_zero_dof_indices
                    elif yaml_zero_dof_indices != zero_dof_indices:
                        raise ValueError("All motion YAML files must use the same zero_dof_indices.")
                elif motion_path.is_file() and motion_path.suffix == ".npz":
                    motion_specs.append((str(motion_path), 1.0, None))

        assert len(motion_specs) > 0, f"No valid motion data found in: {motion_file}"
        if condition_dim > 0 and any(command is None for _, _, command in motion_specs):
            raise ValueError("Every motion requires a [vx, vy, wz] command label for conditional AMP.")
        print("\n=========== AMP Motion File List ===========")
        for idx, (path, weight, command) in enumerate(motion_specs):
            print(f"{idx + 1:2d}. weight={weight:g} command={command}  {path}")
        print(f"=========== Total: {len(motion_specs)} files ===========\n")

        fps_list: list[float] = []
        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        base_lin_vel_b_list: list[torch.Tensor] = []
        base_ang_vel_b_list: list[torch.Tensor] = []
        projected_gravity_b_list: list[torch.Tensor] = []
        motion_weights: list[float] = []
        motion_lengths: list[int] = []
        motion_commands: list[tuple[float, float, float]] = []
        expected_fps: float | None = None
        expected_dof_count: int | None = None
        loaded_motion_paths: list[str] = []

        for file_path, motion_weight, motion_command in motion_specs:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    required_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
                    missing_keys = required_keys.difference(data.files)
                    if missing_keys:
                        raise KeyError(f"Missing required arrays: {sorted(missing_keys)}")

                    fps = float(data["fps"])
                    root_pos = np.asarray(data["root_pos"])
                    root_rot = np.asarray(data["root_rot"])
                    dof_pos = np.asarray(data["dof_pos"])
                    _validate_compact_motion(file_path, fps, root_pos, root_rot, dof_pos)
                    if expected_fps is None:
                        expected_fps = fps
                    elif not math.isclose(fps, expected_fps, rel_tol=1.0e-6):
                        raise ValueError(f"FPS mismatch: expected {expected_fps}, got {fps}")

                    if dof_indices is not None:
                        if max(dof_indices) >= dof_pos.shape[1]:
                            raise ValueError(
                                f"dof_indices require input index {max(dof_indices)}, "
                                f"but {file_path} only has {dof_pos.shape[1]} DOFs."
                            )
                        dof_pos = dof_pos[:, dof_indices]
                    if expected_dof_count is None:
                        expected_dof_count = dof_pos.shape[1]
                    elif dof_pos.shape[1] != expected_dof_count:
                        raise ValueError(
                            f"DOF count mismatch: expected {expected_dof_count}, got {dof_pos.shape[1]}"
                        )

                    joint_pos = torch.as_tensor(dof_pos, dtype=torch.float32, device=device).clone()
                    if zero_dof_indices:
                        joint_pos[:, zero_dof_indices] = 0.0
                    dt = 1.0 / fps
                    joint_vel = _forward_difference(joint_pos, dt)

                    root_pos_w = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
                    # Compact GMR files store quaternions as xyzw; Isaac Lab uses wxyz.
                    root_quat_w = torch.as_tensor(root_rot[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
                    root_quat_w = root_quat_w / torch.linalg.vector_norm(
                        root_quat_w, dim=1, keepdim=True
                    ).clamp_min(1.0e-8)
                    root_lin_vel_w = _forward_difference(root_pos_w, dt)
                    root_ang_vel_w = _angular_velocity_world(root_quat_w, dt)
                    base_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
                    base_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
                    gravity_w = torch.zeros_like(root_pos_w)
                    gravity_w[:, 2] = -1.0
                    projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)

                    fps_list.append(fps)
                    joint_pos_list.append(joint_pos)
                    joint_vel_list.append(joint_vel)
                    base_lin_vel_b_list.append(base_lin_vel_b)
                    base_ang_vel_b_list.append(base_ang_vel_b)
                    projected_gravity_b_list.append(projected_gravity_b)
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
        self.base_lin_vel_b = torch.cat(base_lin_vel_b_list, dim=0)
        self.base_ang_vel_b = torch.cat(base_ang_vel_b_list, dim=0)
        self.projected_gravity_b = torch.cat(projected_gravity_b_list, dim=0)
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
        self.history_length = history_length
        self.include_joint_vel = include_joint_vel
        self.include_base_lin_vel = include_base_lin_vel
        self.include_projected_gravity = include_projected_gravity
        self.include_foot_features = include_foot_features
        self.flatten_history_dim = flatten_history_dim
        self.condition_dim = condition_dim

    def _gather_frame_features(self, idxs: torch.Tensor) -> list[torch.Tensor]:
        """Collect one physical AMP frame in the same order as the environment."""
        features = [self.base_ang_vel_b[idxs], self.joint_pos[idxs]]
        if self.include_joint_vel:
            features.append(self.joint_vel[idxs])
        if self.include_base_lin_vel:
            features.append(self.base_lin_vel_b[idxs])
        if self.include_projected_gravity:
            features.append(self.projected_gravity_b[idxs])
        return features

    def _gather_history(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Gather an oldest-to-newest history without crossing clip boundaries."""
        history_offsets = torch.arange(
            -(self.history_length - 1),
            1,
            device=self.joint_pos.device,
        )
        motion_starts = self.frame_motion_start[batch_indices]
        history = []
        for offset in history_offsets:
            idxs = torch.maximum(batch_indices + offset, motion_starts)
            history.append(torch.cat(self._gather_frame_features(idxs), dim=-1))
        history = torch.stack(history, dim=1)
        if self.flatten_history_dim:
            return history.flatten(start_dim=1)
        return history

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

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` expert histories, weighted by the YAML clip weights.

        The standalone discriminator update needs a batch of an arbitrary size, whereas
        :meth:`mini_batch_generator` ties its batch size to the dataset length.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        batch_indices = torch.multinomial(
            self.frame_sampling_weights,
            num_samples=num_samples,
            replacement=True,
        )
        batch = self._gather_history(batch_indices)
        if self.condition_dim > 0:
            commands = self.motion_commands[self.frame_motion_id[batch_indices]]
            batch = torch.cat([batch, commands], dim=-1)
        return batch

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
