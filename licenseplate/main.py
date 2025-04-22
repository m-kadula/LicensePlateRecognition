import signal
from time import sleep
from typing import Optional, Any
from pathlib import Path
from argparse import ArgumentParser
import string

import yaml
from pydantic import BaseModel

from . import base
from . import action
from . import detection
from . import preprocessor


class CameraConfig(BaseModel):
    camera_interface: str
    kwargs: Optional[dict[str, Any]] = None

    class _DefaultCameraArgs(BaseModel):
        device: int = 0

    class _RaspberryCameraArgs(BaseModel):
        height: int = 1920
        width: int = 1080
        buffer_count: int = 4

    def make(self) -> base.CameraInterface:
        kwargs = self.kwargs if self.kwargs is not None else {}
        if self.camera_interface.strip() == "default":
            kwargs_parsed = self._DefaultCameraArgs.model_validate(kwargs)
            from .camera.default import DefaultCameraInterface
            return DefaultCameraInterface(kwargs_parsed.device)
        elif self.camera_interface.strip() == "raspberry":
            kwargs_parsed = self._RaspberryCameraArgs.model_validate(kwargs)
            from .camera.raspberry import RaspberryCameraInterface
            return RaspberryCameraInterface((kwargs_parsed.height, kwargs_parsed.width), kwargs_parsed.buffer_count)
        else:
            raise ValueError("Available camera interfaces: [default, raspberry]")


class LocalSaveCameraConfig(BaseModel):
    camera: CameraConfig
    max_fps: int = 30
    show_debug_boxes: Optional[bool] = None
    log_cropped_plates: Optional[bool] = None
    log_augmented_plates: Optional[bool] = None

    def make(self, name: str, detection_model: detection.YoloPlateDetectionModel) -> action.LocalSaveManagerArguments:
        return action.LocalSaveManagerArguments(
            name=name,
            detection_model=detection_model,
            camera=self.camera.make(),
            max_fps=self.max_fps,
            show_debug_boxes=self.show_debug_boxes if self.show_debug_boxes is not None else False,
            log_cropped_plates=self.log_cropped_plates if self.log_cropped_plates is not None else False,
            log_augmented_plates=self.log_augmented_plates if self.log_augmented_plates is not None else False,
        )


class LocalSaveConfig(BaseModel):
    yolo_weights_path: str
    original_preprocessor: str
    plate_preprocessor: str
    text_allow_list: str | None
    required_confidence: float = 0.5
    logging_root: str
    cameras: dict[str, LocalSaveCameraConfig]

    def make(self) -> action.LocalSaveManager:
        detection_model = detection.YoloPlateDetectionModel(
            yolo_weights_path=Path(self.yolo_weights_path).resolve(),
            original_frame_preprocessor=get_preprocessor(self.original_preprocessor),
            license_plate_preprocessor=get_preprocessor(self.plate_preprocessor),
            text_allow_list=self.text_allow_list,
            required_confidence=self.required_confidence,
        )
        parsed_cameras = [camera.make(name, detection_model) for name, camera in self.cameras.items()]
        return action.LocalSaveManager(
            cameras=parsed_cameras,
            logging_root=Path(self.logging_root).resolve()
        )


class Config(BaseModel):
    instances: dict[str, LocalSaveConfig]

    def make(self) -> dict[str, base.ManagerInterface]:
        return {key: value.make() for key, value in self.instances.items()}


def get_preprocessor(name: str) -> base.preprocessor_type:
    if name == 'identity':
        return preprocessor.preprocess_identity
    elif name == 'black_and_white':
        return preprocessor.preprocess_black_on_white
    else:
        raise ValueError("Preprocessors allowed: [identity, black_and_white].")


example_config = Config(
    instances={
        "instance1": LocalSaveConfig(
            yolo_weights_path=str(Path.cwd() / 'weights.pt'),
            original_preprocessor='identity',
            plate_preprocessor='black_and_white',
            text_allow_list=string.ascii_uppercase + string.digits,
            required_confidence=0.5,
            logging_root=str(Path.cwd() / 'detected'),
            cameras={
                "camera1": LocalSaveCameraConfig(
                    camera=CameraConfig(
                        camera_interface="default",
                        kwargs={"device": 0},
                    ),
                    max_fps=30,
                    show_debug_boxes=True,
                    log_cropped_plates=True,
                    log_augmented_plates=True,
                )
            }
        )
    }
)


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
            yaml.dump(example_config.model_dump(), f)

    elif args.command == "run":
        with open(args.configuration_file) as f:
            data = yaml.load(f, yaml.SafeLoader)
        global_config = Config.model_validate(data)

        managers = global_config.make()

        for manager in managers.values():
            manager.start()

        def interrupt_handler(signum, frame):
            for m in managers.values():
                m.stop()
            exit(0)

        signal.signal(signal.SIGINT, interrupt_handler)
        signal.signal(signal.SIGTERM, interrupt_handler)

        while True:
            sleep(1)


if __name__ == "__main__":
    main()
