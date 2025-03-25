import importlib
from typing import Any, Optional
from pathlib import Path
from string import ascii_uppercase, digits
from argparse import ArgumentParser

import yaml
from pydantic import BaseModel

from .detection import PlateDetectionModel
from .camera.base import CameraInterface
from .action.base import ActionInterface, BaseActionManager
from .preprocessor.base import PreprocessorInterface


# pydantic config models


class InterfaceConfig(BaseModel):
    which: str
    kwargs: Optional[dict[str, Any]] = None


class LoopConfig(BaseModel):
    yolo_weights_path: str
    general_preprocessor: InterfaceConfig
    license_plate_preprocessor: InterfaceConfig
    camera_interface: InterfaceConfig
    action_interface: InterfaceConfig
    text_allow_list: Optional[str] = None
    required_confidence: float
    max_fps: int


class ManagerConfig(BaseModel):
    which: str
    apply_to: list[InterfaceConfig]  # which instances to apply to
    kwargs: Optional[dict[str, Any]] = None


class GlobalConfig(BaseModel):
    instances: dict[str, LoopConfig]
    managers: Optional[dict[str, ManagerConfig]] = None


# example

example = GlobalConfig(
    instances={
        "camera1": LoopConfig(
            yolo_weights_path=str(Path(__file__).parents[1] / "weights.pt"),
            general_preprocessor=InterfaceConfig(
                which=".base.IdentityPreprocessor",
                kwargs=None,
            ),
            license_plate_preprocessor=InterfaceConfig(
                which=".polish_plate.PolishLicensePlatePreprocessor",
                kwargs=None,
            ),
            camera_interface=InterfaceConfig(
                which=".default.DefaultCameraInterface",
                kwargs={"device": 0},
            ),
            action_interface=InterfaceConfig(
                which=".localsave.LocalSave",
                kwargs={
                    "show_debug_boxes": True,
                },
            ),
            text_allow_list=ascii_uppercase + digits,
            required_confidence=0.5,
            max_fps=30,
        )
    },
    managers={
        "manager1": ManagerConfig(
            which=".localsave.LocalSaveManager",
            apply_to=[InterfaceConfig(which="camera1")],
            kwargs={
                "logging_path": str(Path(__file__).parents[1] / "detected")
            }
        )
    },
)


def dynamic_import_class(package: str, module_path: str):
    *path, class_name = module_path.split(".")
    path_str = ".".join(path)

    module = importlib.import_module(package=package, name=path_str)
    return getattr(module, class_name)


def make_class_instance(package: str, config: InterfaceConfig | ManagerConfig):
    class_ = dynamic_import_class(__package__ + package, config.which)
    kwargs = config.kwargs if config.kwargs is not None else {}
    return class_.get_instance(kwargs)


def instance_check(expected: type, got: Any):
    if not isinstance(got, expected):
        raise TypeError(f"Expected type {expected}, got {type(got)}.")


def configure_action(
    loop_config: LoopConfig,
) -> ActionInterface:
    general_preprocessor: PreprocessorInterface = make_class_instance(
        ".preprocessor", loop_config.general_preprocessor
    )
    instance_check(PreprocessorInterface, general_preprocessor)
    license_plate_preprocessor: PreprocessorInterface = make_class_instance(
        ".preprocessor", loop_config.license_plate_preprocessor
    )
    instance_check(PreprocessorInterface, license_plate_preprocessor)
    camera: CameraInterface = make_class_instance(
        ".camera", loop_config.camera_interface
    )
    instance_check(CameraInterface, camera)

    detection_model = PlateDetectionModel(
        yolo_weights_path=Path(loop_config.yolo_weights_path).resolve(),
        original_frame_preprocessor=general_preprocessor,
        license_plate_preprocessor=license_plate_preprocessor,
        text_allow_list=loop_config.text_allow_list,
        required_confidence=loop_config.required_confidence,
    )

    action_class: type[ActionInterface] = dynamic_import_class(
        __package__ + ".action", loop_config.action_interface.which
    )
    action_kwargs = loop_config.action_interface.kwargs if loop_config.action_interface.kwargs is not None else {}
    action = action_class.get_instance(
        detection_model, camera, loop_config.max_fps, action_kwargs
    )

    return action


def configure_manager(
    instances: dict[str, ActionInterface], manager_config: ManagerConfig
) -> BaseActionManager:
    manager: BaseActionManager = make_class_instance(".action", manager_config)
    instance_check(BaseActionManager, manager)

    for instance in manager_config.apply_to:
        if instance.which not in instances.keys():
            raise ValueError(f"No instance with name {instance.which} defined.")
        kwargs = instance.kwargs if instance.kwargs is not None else {}
        manager.register_camera(instance.which, instances[instance.which], kwargs)
    manager.finish_registration()

    return manager


def configure(
    config: GlobalConfig,
) -> dict[str, BaseActionManager]:
    all_instances: dict[str, ActionInterface] = {}
    for name, loop_config in config.instances.items():
        action = configure_action(loop_config)
        all_instances[name] = action

    all_managers: dict[str, BaseActionManager] = {}
    if config.managers is not None:
        for name, manager_config in config.managers.items():
            manager = configure_manager(all_instances, manager_config)
            all_managers[name] = manager

    default_manager = BaseActionManager()
    all_managers['__default__'] = default_manager

    for name, instance in all_instances.items():
        if instance.manager is None:
            default_manager.register_camera(name, instance, {})
    default_manager.finish_registration()

    return all_managers


def main():
    parser = ArgumentParser("Daemon for license plate recognition")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    run_subparser = subparsers.add_parser("run", help="run the daemon")
    run_subparser.add_argument(
        "configuration_file", type=Path, help="Configuration file."
    )

    generate_subparser = subparsers.add_parser(
        "generate", help="Generate an example config file."
    )
    generate_subparser.add_argument(
        "file_name", type=Path, help="File to save config to."
    )

    args = parser.parse_args()

    if args.command == "generate":
        if args.file_name.exists():
            raise FileExistsError(f"File {args.file_name} already exists.")
        with open(args.file_name, "w") as f:
            yaml.dump(example.model_dump(), f)

    elif args.command == "run":
        with open(args.configuration_file) as f:
            data = yaml.load(f, yaml.SafeLoader)
        global_config = GlobalConfig.model_validate(data)

        managers = configure(global_config)

        for manager in managers.values():
            manager.start()

        try:
            while True:
                input()
        except KeyboardInterrupt:
            for manager in managers.values():
                manager.stop()


if __name__ == "__main__":
    main()
