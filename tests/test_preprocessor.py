import sys
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import cv2

from licenseplate.base import preprocessor_type
from licenseplate import preprocessor as prc


preprocessor_choices: dict[str, preprocessor_type] = {
    "identity": prc.preprocess_identity,
    "black-white": prc.preprocess_black_on_white,
}


def main():
    parser = ArgumentParser("Run a selected preprocessor on all images in a directory.")
    parser.add_argument("preprocessor", choices=list(preprocessor_choices.keys()))
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "test_results"
        / ("preprocessor_test-" + datetime.now().isoformat()),
    )
    args = parser.parse_args()

    preprocessor: preprocessor_type = preprocessor_choices[args.preprocessor]
    images_path: Path = args.directory.resolve()
    output_path: Path = args.output.resolve()

    if not images_path.exists() or not images_path.is_dir():
        print(
            "Error: Path provided does not exists or is not a directory.",
            file=sys.stderr,
        )
        exit(1)

    if output_path.exists():
        print("Output directory already exists.", file=sys.stderr)
        exit(1)

    output_path.mkdir()

    for file in filter(lambda pth: pth.suffix == ".jpg", images_path.iterdir()):
        assert isinstance(file, Path)
        img = cv2.imread(str(file))
        altered_img = preprocessor(img)
        output_file_path = output_path / file.name
        cv2.imwrite(str(output_file_path), altered_img)

    exit(0)


if __name__ == "__main__":
    main()
