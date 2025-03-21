import importlib
import sys
from typing import Any, Callable
from pathlib import Path
from string import ascii_uppercase, digits
from argparse import ArgumentParser
import threading

import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from .loop import detection_loop
from .detection import PlateDetectionModel
from .camera.base import CameraInterface
from .action.base import ActionInterface
from .logger import get_logger


# pydantic config models

class LoopConfig(BaseModel):
    yolo_weights_path: str
    general_preprocessor: str
    license_plate_preprocessor: str
    text_allow_list: str
    plate_regex: str
    required_confidence: float
    camera_interface: str
    action_interface: str
    max_fps: int

    args: dict[str, list[Any]]
    kwargs: dict[str, dict[str, Any]]
    env: dict[str, Any]

class GlobalConfig(BaseModel):
    instances: dict[str, LoopConfig]

# example

example = GlobalConfig(
    instances={
        "camera1": LoopConfig(
            yolo_weights_path=str(Path(__file__).parents[1] / 'weights.pt'),
            general_preprocessor='preprocess_identity',
            license_plate_preprocessor='preprocess_polish_license_plate',
            text_allow_list=ascii_uppercase + digits,
            plate_regex=r"[A-Z]{1,3} ?[0-9A-Z]{3,5}",
            required_confidence=0.5,
            camera_interface='.macos.MacOSCameraInterface',
            action_interface='.localsave.LocalSaveInterface',
            max_fps=30,

            args={
                'camera_interface': [0],
                'action_interface': [str(Path(__file__).parents[1] / 'detected'), True]
            },

            kwargs={
                'camera_interface': {},
                'action_interface': {}
            },

            env={}
        )
    }
)

def dynamic_import_class(package: str, module_path: str):
    *path, class_name = module_path.split('.')
    path = '.'.join(path)

    module = importlib.import_module(package=package, name=path)
    return getattr(module, class_name)


def configure_loop(
    loop_config: LoopConfig
) -> tuple[PlateDetectionModel, CameraInterface, ActionInterface, Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
    module = importlib.import_module(package=__package__, name=f".preprocessors")
    general_preprocessor = getattr(module, loop_config.general_preprocessor)

    module = importlib.import_module(package=__package__, name=f".preprocessors")
    license_plate_preprocessor = getattr(module, loop_config.license_plate_preprocessor)

    camera_class = dynamic_import_class(__package__ + '.camera', loop_config.camera_interface)
    action_class = dynamic_import_class(__package__ + '.action', loop_config.action_interface)

    camera_args = loop_config.args.get('camera_interface', [])
    camera_kwargs = loop_config.kwargs.get('camera_interface', {})

    camera = camera_class(*camera_args, **camera_kwargs)

    action_args = loop_config.args.get('action_interface', [])
    action_kwargs = loop_config.kwargs.get('action_interface', {})

    action = action_class(*action_args, **action_kwargs)

    detection_model = PlateDetectionModel(
        yolo_weights_path=Path(loop_config.yolo_weights_path).resolve(),
        original_frame_preprocessor=general_preprocessor,
        license_plate_preprocessor=license_plate_preprocessor,
        text_allow_list=loop_config.text_allow_list,
        plate_regex=loop_config.plate_regex,
        required_confidence=loop_config.required_confidence
    )

    return detection_model, camera, action, general_preprocessor, license_plate_preprocessor


def main():
    parser = ArgumentParser("Daemon for license plate recognition")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")

    run_subparser = subparsers.add_parser("run", help="run the daemon")
    run_subparser.add_argument("configuration_file", type=Path, help="Configuration file.")

    generate_subparser = subparsers.add_parser('generate', help="Generate an example config file.")
    generate_subparser.add_argument('file_name', type=Path, help="File to save config to.")

    args = parser.parse_args()

    if args.command == 'generate':
        if args.file_name.exists():
            raise FileExistsError(f"File {args.file_name} already exists.")
        with open(args.file_name, 'w') as f:
            yaml.dump(example.model_dump(), f)

    elif args.command == 'run':
        with open(args.configuration_file) as f:
            data = yaml.load(f, yaml.Loader)
        global_config = GlobalConfig.model_validate(data)

        threads = []
        for name, loop_config in global_config.instances.items():
            detection_model, camera, action, base_processor, plate_processor = configure_loop(loop_config)
            thread = threading.Thread(target=detection_loop, args=[
                detection_model,
                camera,
                action,
                get_logger(name, sys.stdout),
                loop_config.max_fps
            ])
            threads.append(thread)
            thread.start()

        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
