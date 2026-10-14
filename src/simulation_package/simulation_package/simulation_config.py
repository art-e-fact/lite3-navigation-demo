from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, get_type_hints

try:
	import yaml
except ImportError:  # pragma: no cover - reported when config loading is used
	yaml = None

try:
	from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - source-tree fallback handles this
	get_package_share_directory = None


PACKAGE_NAME = "simulation_package"
LEGACY_PACKAGE_NAME = "lite3_sdk_deploy"
ASSETS_DIR_NAME = "assets"
DEFAULT_SCENE_URI = "package://simulation_package/assets/lite3_mjcf/mjcf/stairs_floors.xml"
DEFAULT_ROBOT_DESCRIPTION_URI = "package://simulation_package/assets/lite3_mjcf/mjcf/Lite3.xml"
M20_ROBOT_DESCRIPTION_URI = "package://simulation_package/assets/m20_mjcf/mjcf/M20.xml"
PACKAGE_URI_PREFIX = "package://"
# File suffixes accepted as MuJoCo MJCF scene/robot descriptions.
MJCF_SUFFIXES = {".xml", ".mjcf"}
HF_URI_PREFIX = "hf://"
# Hub web URLs are accepted too, so a link copied from the browser works as-is.
HF_URI_PREFIXES = (HF_URI_PREFIX, "https://huggingface.co/", "https://hf.co/")
HF_DOWNLOADABLE_REPO_TYPES = {"model", "dataset", "space"}

ROBOT_PRESETS: dict[str, dict[str, str | float]] = {
	"lite3": {
		"name": "Lite3",
		"robot_description": DEFAULT_ROBOT_DESCRIPTION_URI,
		"base_height": 0.43,
	},
	"m20": {
		"name": "M20",
		"robot_description": M20_ROBOT_DESCRIPTION_URI,
		"base_height": 1.0,
	},
}


@dataclass
class Lidar2DConfig:
	enabled: bool = False
	site_name: str = "lidar_site"
	frame_id: str = "lidar"
	topic: str = "/scan"
	frequency_hz: float = 10.0
	num_rays: int = 360
	range_min: float = 0.15
	range_max: float = 8.0
	visualize_rays: bool = False


@dataclass
class FollowCameraConfig:
	enabled: bool = False
	video_path: str = ""
	fps: float = 20.0
	width: int = 800
	height: int = 600
	distance_m: float = 4.0
	elevation_deg: float = -18.0
	azimuth_offset_deg: float = 60.0
	target_height_m: float = 0.55
	smoothing: float = 0.08
	quality: int = 8
	# Vertical field of view. MuJoCo's default free camera uses fovy=45, so both
	# backends frame the robot identically at the same distance_m.
	fov_deg: float = 45.0
	# Turn with the robot (a chase camera) or keep a fixed heading (an overhead camera:
	# elevation_deg -89, follow_yaw false keeps the map's orientation).
	follow_yaw: bool = True


@dataclass
class SensorsConfig:
	lidar_2d: Lidar2DConfig = field(default_factory=Lidar2DConfig)
	follow_camera: FollowCameraConfig = field(default_factory=FollowCameraConfig)
	# A second video, same options, rendered off its own offscreen viewer.
	overhead_camera: FollowCameraConfig = field(default_factory=FollowCameraConfig)


@dataclass
class RerunConfig:
	enabled: bool = False
	spawn: bool = True
	save_path: str = ""
	# Terminate the viewer process spawned by this simulator when the simulator exits.
	# The viewer is spawned detached (own session), so process-group cleanup by a test
	# harness cannot reach it; tests set this to true to avoid leaking viewer windows.
	close_viewer_on_exit: bool = False


@dataclass
class StartPoseConfig:
	"""Where the robot starts, in the scene's frame (metres, yaw in radians).

	A field left unset keeps the default start (DEFAULT_ROBOT_START_POSE). Set from a config
	file (robot: {start_pose: {x: 1.0, y: -0.5, yaw: 1.57}}) or on the command
	line (--set robot.start_pose.x=1.0), e.g. from a recorded route's start.
	"""
	x: float | None = None
	y: float | None = None
	yaw: float | None = None

	def apply(self, pose: tuple[float, float, float]) -> tuple[float, float, float]:
		"""``pose`` with the fields that are set here replaced."""
		x, y, yaw = pose
		return (
			x if self.x is None else float(self.x),
			y if self.y is None else float(self.y),
			yaw if self.yaw is None else float(self.yaw),
		)


@dataclass
class RobotConfig:
	model: str = "lite3"
	robot_description: str = DEFAULT_ROBOT_DESCRIPTION_URI
	state_frequency_hz: float = 50.0
	start_pose: StartPoseConfig = field(default_factory=StartPoseConfig)

	def preset(self) -> dict[str, str | float]:
		key = self.model.strip().lower()
		if key not in ROBOT_PRESETS:
			raise ValueError(f"Unknown robot.model '{self.model}', expected one of: {', '.join(sorted(ROBOT_PRESETS))}")
		return ROBOT_PRESETS[key]

	@property
	def model_name(self) -> str:
		return str(self.preset()["name"])


@dataclass
class SimulationConfig:
	simulator: str = "newton"
	scene: str = DEFAULT_SCENE_URI
	robot: RobotConfig = field(default_factory=RobotConfig)
	headless: bool = True
	sensors: SensorsConfig = field(default_factory=SensorsConfig)
	rerun: RerunConfig = field(default_factory=RerunConfig)

	@classmethod
	def load(cls, config_path: str | None = None) -> "SimulationConfig":
		data = _dataclass_to_dict(cls())
		if config_path:
			data = _deep_merge(data, _read_yaml(resolve_path(config_path)))
		return cls.from_dict(data)

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "SimulationConfig":
		config = _dataclass_from_dict(cls, data or {})
		preset = config.robot.preset()
		config.robot.robot_description = str(preset["robot_description"])
		if config.headless and config.rerun.spawn:
			return config.with_overrides({"rerun.spawn": False})
		return config

	def with_overrides(self, overrides: dict[str, Any]) -> "SimulationConfig":
		data = _dataclass_to_dict(self)
		data = _deep_merge(data, _unflatten(overrides))
		return self.from_dict(data)

	def resolved_scene(self) -> str | None:
		raw = str(self.scene).strip()
		if not raw:
			return None
		return str(resolve_path(raw))

	def resolved_robot_description(self) -> str:
		return str(resolve_path(self.robot.robot_description))

	def validate(self) -> list[str]:
		errors: list[str] = []
		simulator = self.simulator.lower()
		if simulator != "newton":
			errors.append("simulator must be 'newton'")

		try:
			self.robot.preset()
		except ValueError as exc:
			errors.append(str(exc))

		scene_value = str(self.scene).strip()
		scene_path = None
		if scene_value:
			scene_path = resolve_path(scene_value, must_exist=False)
		robot_path = resolve_path(self.robot.robot_description, must_exist=False)
		if scene_path is not None and scene_path.suffix.lower() not in MJCF_SUFFIXES:
			errors.append("scene must be an MJCF/XML file")
		if robot_path.suffix.lower() == ".usd" and simulator != "newton":
			errors.append("USD robot_description is only supported by the Newton simulator")
		if robot_path.suffix.lower() not in MJCF_SUFFIXES | {".usd"}:
			errors.append("robot_description must be an MJCF/XML or USD file")
		if scene_path is not None and not scene_path.exists():
			errors.append(f"scene does not exist: {scene_path}")
		if not robot_path.exists():
			errors.append(f"robot_description does not exist: {robot_path}")

		_validate_positive(errors, "robot.state_frequency_hz", self.robot.state_frequency_hz)
		for axis in ("x", "y", "yaw"):
			value = getattr(self.robot.start_pose, axis)
			if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
				errors.append(f"robot.start_pose.{axis} must be a number")

		sensors = self.sensors
		_validate_positive(errors, "sensors.lidar_2d.frequency_hz", sensors.lidar_2d.frequency_hz)
		_validate_positive(errors, "sensors.lidar_2d.num_rays", sensors.lidar_2d.num_rays)
		_validate_range(errors, "sensors.lidar_2d", sensors.lidar_2d.range_min, sensors.lidar_2d.range_max)
		for name, camera in (("follow_camera", sensors.follow_camera), ("overhead_camera", sensors.overhead_camera)):
			if camera.enabled and not str(camera.video_path).strip():
				errors.append(f"sensors.{name}.video_path must be set when camera recording is enabled")
			_validate_positive(errors, f"sensors.{name}.fps", camera.fps)
			_validate_positive(errors, f"sensors.{name}.width", camera.width)
			_validate_positive(errors, f"sensors.{name}.height", camera.height)
			_validate_positive(errors, f"sensors.{name}.distance_m", camera.distance_m)
			_validate_positive(errors, f"sensors.{name}.target_height_m", camera.target_height_m)
			_validate_positive(errors, f"sensors.{name}.quality", camera.quality)
			_validate_positive(errors, f"sensors.{name}.fov_deg", camera.fov_deg)
			_validate_unit_interval(errors, f"sensors.{name}.smoothing", camera.smoothing)

		return errors

def candidate_package_roots() -> list[Path]:
	roots: list[Path] = []
	seen: set[Path] = set()

	if get_package_share_directory is not None:
		for package_name in (PACKAGE_NAME, LEGACY_PACKAGE_NAME):
			try:
				package_root = Path(get_package_share_directory(package_name)).resolve()
				if package_root not in seen:
					roots.append(package_root)
					seen.add(package_root)
			except Exception:
				pass

	for parent in Path(__file__).resolve().parents:
		candidates = (
			parent,
			parent / "share" / PACKAGE_NAME,
			parent / "share" / LEGACY_PACKAGE_NAME,
		)
		for candidate in candidates:
			resolved = candidate.resolve()
			if resolved in seen:
				continue
			if (
				(resolved / ASSETS_DIR_NAME).exists()
				or (resolved / "Lite3_description").exists()
				or (resolved / "config").exists()
			):
				roots.append(resolved)
				seen.add(resolved)
	if not roots:
		roots.append(Path(__file__).resolve().parents[3])
	return roots


def _resource_search_roots() -> list[Path]:
	bases: list[Path] = []
	seen: set[Path] = set()
	for root in candidate_package_roots():
		for base in (
			root,
			root / ASSETS_DIR_NAME,
			root / "Lite3_description",
			root / "config",
		):
			resolved = base.resolve()
			if resolved in seen:
				continue
			bases.append(resolved)
			seen.add(resolved)
	return bases


@lru_cache(maxsize=None)
def _resolve_hf_uri(raw: str) -> Path:
	"""Download a Hugging Face resource and return its local path.

	Accepts the canonical hf:// form as well as Hub web URLs, for example
	hf://datasets/art-e-fact/karuizawa-worlds@abc1234/karuizawa/part1/scene.xml
	"""
	try:
		from huggingface_hub import parse_hf_uri, snapshot_download
	except ImportError as exc:  # pragma: no cover - depends on environment
		raise RuntimeError(
			f"Resolving '{raw}' requires the huggingface_hub package. "
			"Install it in the simulation environment to use hf:// resources."
		) from exc

	uri = parse_hf_uri(raw)
	if uri.type not in HF_DOWNLOADABLE_REPO_TYPES:
		raise ValueError(
			f"Unsupported Hugging Face URI '{raw}', expected one of "
			f"{sorted(HF_DOWNLOADABLE_REPO_TYPES)}, got '{uri.type}'"
		)
	# Scene XMLs reference their meshes relative to their own directory, so fetch
	# the whole containing folder instead of just the requested file.
	parent = Path(uri.path_in_repo).parent
	allow_patterns = None if parent == Path(".") else [f"{parent.as_posix()}/*"]
	snapshot = snapshot_download(
		repo_id=uri.id,
		repo_type=uri.type,
		revision=uri.revision,
		allow_patterns=allow_patterns,
	)
	# Deliberately not resolve(): entries in the snapshot directory are symlinks
	# into the content-addressed blob store, where files carry no extension and
	# have no siblings. The scene must keep its suffix and stay next to its meshes.
	return Path(snapshot) / uri.path_in_repo


def resolve_path(path_value: str | os.PathLike[str], must_exist: bool = False) -> Path:
	raw = str(path_value).strip()
	if raw.startswith(HF_URI_PREFIXES):
		return _resolve_hf_uri(raw)
	if raw.startswith(PACKAGE_URI_PREFIX):
		package_and_path = raw[len(PACKAGE_URI_PREFIX) :]
		package, _, rel = package_and_path.partition("/")
		if package.lower() not in {PACKAGE_NAME, LEGACY_PACKAGE_NAME}:
			raise ValueError(
				f"Unsupported package URI '{raw}', expected package://{PACKAGE_NAME}/..."
			)
		rel_path = Path(rel)
		fallback = None
		for root in _resource_search_roots():
			candidate = (root / rel_path).resolve()
			if candidate.exists():
				return candidate
			if fallback is None:
				fallback = candidate
		if fallback is not None and not must_exist:
			return fallback

	candidate = Path(raw).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	if candidate.exists():
		return candidate.resolve()
	fallback = None
	for root in _resource_search_roots():
		for base in (
			root,
			root / "lite3_mjcf" / "mjcf",
			root / "m20_mjcf" / "mjcf",
		):
			resolved = (base / candidate).resolve()
			if resolved.exists():
				return resolved
			if fallback is None:
				fallback = resolved
	if fallback is not None and not must_exist:
		return fallback
	return candidate.resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
	if yaml is None:
		raise RuntimeError("PyYAML is required to load simulation configuration")
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as config_file:
		data = yaml.safe_load(config_file) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Simulation config must be a YAML mapping: {path}")
	return data


def _dataclass_from_dict(cls, data: dict[str, Any]):
	kwargs = {}
	field_map = {field.name: field for field in fields(cls)}
	type_hints = get_type_hints(cls)
	for name, field_info in field_map.items():
		value = data.get(name)
		if value is None:
			continue
		field_type = type_hints.get(name, field_info.type)
		if hasattr(field_type, "__dataclass_fields__"):
			kwargs[name] = _dataclass_from_dict(field_type, value)
		else:
			kwargs[name] = value
	return cls(**kwargs)


def _dataclass_to_dict(value):
	if is_dataclass(value):
		return {field_info.name: _dataclass_to_dict(getattr(value, field_info.name)) for field_info in fields(value)}
	if isinstance(value, list):
		return [_dataclass_to_dict(item) for item in value]
	return copy.deepcopy(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	merged = copy.deepcopy(base)
	for key, value in override.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _deep_merge(merged[key], value)
		else:
			merged[key] = copy.deepcopy(value)
	return merged


def _unflatten(overrides: dict[str, Any]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for dotted_key, value in overrides.items():
		current = result
		parts = dotted_key.split(".")
		for part in parts[:-1]:
			current = current.setdefault(part, {})
		current[parts[-1]] = value
	return result


def _validate_positive(errors: list[str], name: str, value: int | float):
	if value <= 0:
		errors.append(f"{name} must be positive")


def _validate_range(errors: list[str], name: str, min_value: float, max_value: float):
	if min_value < 0:
		errors.append(f"{name}_range_min must be non-negative")
	if max_value <= min_value:
		errors.append(f"{name}_range_max must be greater than range_min")


def _validate_unit_interval(errors: list[str], name: str, value: float):
	if value < 0.0 or value > 1.0:
		errors.append(f"{name} must be between 0.0 and 1.0")
