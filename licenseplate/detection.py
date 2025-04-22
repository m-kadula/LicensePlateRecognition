from typing import Optional
from pathlib import Path

from numpy.typing import NDArray
import cv2
import easyocr
from ultralytics import YOLO

from . import base


class LicensePlateFinder:
    def __init__(self, weights_path: Path):
        self.model = YOLO(weights_path)

    def run(self, image: NDArray) -> list[base.FinderResult]:
        result = self.model(image, verbose=False)[0]
        out = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            out.append(base.FinderResult(confidence=confidence, box=(x1, y1, x2, y2)))

        return sorted(out, key=lambda x: x.confidence, reverse=True)

    def __call__(self, image: NDArray) -> list[base.FinderResult]:
        return self.run(image)


class TextExtractor:
    def __init__(self, allow_list: Optional[str] = None):
        self.allow_list = allow_list
        self.reader = easyocr.Reader(["en"])

    def run(self, image: NDArray) -> list[base.ExtractorResult]:
        detected = self.reader.readtext(
            image, allowlist=self.allow_list, decoder="beamsearch"
        )
        out = []

        for bbox, text, confidence in detected:
            box = tuple(map(lambda x: (int(x[0]), int(x[1])), bbox))
            assert len(box) == 4
            out.append(base.ExtractorResult(text=text, confidence=float(confidence), box=box))

        return out

    def __call__(self, image: NDArray) -> list[base.ExtractorResult]:
        return self.run(image)


class YoloPlateDetectionModel(base.PlateDetectionModel):
    def __init__(
        self,
        yolo_weights_path: Path,
        original_frame_preprocessor: base.preprocessor_type,
        license_plate_preprocessor: base.preprocessor_type,
        text_allow_list: Optional[str] = None,
        required_confidence: float = 0.5,
    ):
        self.finder = LicensePlateFinder(yolo_weights_path)
        self.extractor = TextExtractor(text_allow_list)
        self.original_image_preprocessor = original_frame_preprocessor
        self.license_plate_preprocessor = license_plate_preprocessor
        self.required_confidence = required_confidence

    def detect_plates(
        self, image: NDArray
    ) -> base.DetectionResults:
        preprocessed_image = self.original_image_preprocessor(image)
        found_boxes = self.finder(preprocessed_image)
        out = base.DetectionResults(original_image=image, general_preprocessed_image=preprocessed_image, det_results=[])

        for box in found_boxes:
            x1, y1, x2, y2 = box.box
            cropped_image = preprocessed_image[y1:y2, x1:x2]
            altered_image = self.license_plate_preprocessor(cropped_image)
            found_text = self.extractor(altered_image)
            found_text = list(
                filter(lambda x: x.confidence >= self.required_confidence, found_text)
            )

            out.det_results.append(base.SingleDetectionResult(
                cropped_plate_image=cropped_image,
                text_preprocessed_image=altered_image,
                finder_result=box,
                ext_results=found_text
            ))

        return out


def convert_extractor_bbox_to_whole_image(
    finder_bbox_xyxy: tuple[int, int, int, int], extractor_bbox_points: tuple
):
    f_xtl, f_ytl, f_xbr, f_xbr = finder_bbox_xyxy
    func = lambda p: (f_xtl + p[0], f_ytl + p[1])
    return tuple(map(func, extractor_bbox_points))


def visualise_all(result: base.DetectionResults, show_debug_boxes: bool = False) -> NDArray:
    image = result.general_preprocessed_image.copy()
    for detection_result in result.det_results:
        image = visualise(image, detection_result.finder_result, detection_result.ext_results, show_debug_boxes)
    return image


def visualise(
    image: NDArray,
    finder_result: base.FinderResult,
    extractor_results: list[base.ExtractorResult],
    show_debug_boxes=False,
) -> NDArray:
    image = image.copy()
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
