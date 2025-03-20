from pathlib import Path
from dataclasses import dataclass
import re
from string import ascii_uppercase, digits
from typing import Callable

from numpy.typing import NDArray
import cv2
import easyocr
from ultralytics import YOLO


@dataclass
class FinderResult:
    confidence: float
    box: tuple[int, int, int, int]


class LicensePlateFinder:
    def __init__(self, weights_path: Path):
        self.model = YOLO(weights_path)

    def run(self, image: NDArray) -> list[FinderResult]:
        result = self.model(image)[0]
        out = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            out.append(FinderResult(confidence=confidence, box=(x1, y1, x2, y2)))

        return sorted(out, key=lambda x: x.confidence, reverse=True)

    def __call__(self, image: NDArray) -> list[FinderResult]:
        return self.run(image)


@dataclass
class ExtractorResult:
    text: str
    confidence: float
    box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


class TextExtractor:
    def __init__(self, allow_list: str = ascii_uppercase + digits):
        self.allow_list = allow_list
        self.reader = easyocr.Reader(["en"])

    def run(self, image: NDArray) -> list[ExtractorResult]:
        detected = self.reader.readtext(
            image, allowlist=self.allow_list, decoder="beamsearch"
        )
        out = []

        for bbox, text, confidence in detected:
            box = tuple(map(tuple, bbox))
            out.append(ExtractorResult(text=text, confidence=confidence, box=box))

        return out

    def __call__(self, image: NDArray) -> list[ExtractorResult]:
        return self.run(image)


class LicensePlateValidator:
    def __init__(
        self,
        plate_regex: str = r"[A-Z]{1,3} ?[0-9A-Z]{3,5}",
        required_confidence: float = 0.5,
    ):
        self.plate_regex = re.compile(plate_regex)
        self.required_confidence = required_confidence

    def validate(self, extraction: ExtractorResult) -> bool:
        return (
            self.plate_regex.match(extraction.text) is not None
            and extraction.confidence >= self.required_confidence
        )


def fix_plate(extraction: ExtractorResult):
    text = extraction.text
    text = text.replace("O", "0").replace(" ", "").strip()
    extraction.text = text


def convert_extractor_bbox_to_whole_image(
    finder_bbox_xyxy: tuple[int, int, int, int], extractor_bbox_points: tuple
):
    f_xtl, f_ytl, f_xbr, f_xbr = finder_bbox_xyxy
    func = lambda p: (f_xtl + p[0], f_ytl + p[1])
    return tuple(map(func, extractor_bbox_points))


def detect_plates(
    image: NDArray,
    finder: LicensePlateFinder,
    extractor: TextExtractor,
    preprocessor: Callable[[NDArray], NDArray],
    validator: LicensePlateValidator,
) -> list[tuple[FinderResult, list[ExtractorResult]]]:
    found_boxes = finder(image)
    out = []

    for box in found_boxes:
        x1, y1, x2, y2 = box.box
        cropped_image = image[y1:y2, x1:x2]
        altered_image = preprocessor(cropped_image)
        found_text = extractor(altered_image)
        for f in found_text:
            fix_plate(f)
        found_text = list(filter(lambda x: validator.validate(x), found_text))

        out.append((box, found_text))

    return out


def visualise(
    image: NDArray,
    results: list[tuple[FinderResult, list[ExtractorResult]]],
    show_debug_boxes=False,
) -> NDArray:
    image = image.copy()
    for finder_result, extractor_results in results:
        if show_debug_boxes:
            debug_box = finder_result.box
            cv2.rectangle(
                image,
                (debug_box[0], debug_box[1]),
                (debug_box[2], debug_box[3]),
                (255, 0, 0),
                2,
            )
        for extractor_result in extractor_results:
            box = convert_extractor_bbox_to_whole_image(
                finder_result.box, extractor_result.box
            )
            confidence = extractor_result.confidence
            text = extractor_result.text

            top_left, _, bottom_right, _ = box
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{text} ({confidence:.2f})",
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    return image


# if __name__ == '__main__':
#     engine = Path(__file__).parents[1] / 'runs/detect/train/weights/best.pt'
#     image = Path(__file__).parents[1] / 'dataset/images/val/2.jpg'
#     img = cv2.imread(str(image))
#
#     finder = YoloLicensePlateFinder(engine)
#     extractor = TextExtractor()
#
#     detected = detect_plates(img, finder, extractor)
#     print(detected)
#     vis = visualise(img, detected)
#
#     cv2.imshow("OCR Detection", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
