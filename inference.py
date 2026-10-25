from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import lru_cache
import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable
import shutil
import uuid

import cv2
import numpy as np
from rapidocr import RapidOCR
from ultralytics import YOLO

import inference_config as config


_PROCESSING_LOCK = Lock()
_MODEL_FILENAMES = {
    "vehicle": config.VEHICLE_MODEL_NAME,
    "helmet": config.HELMET_MODEL_NAME,
    "plate": config.PLATE_MODEL_NAME,
}
_DIRECTION_VECTORS = {
    "TOP_TO_BOTTOM": (0.0, 1.0),
    "BOTTOM_TO_TOP": (0.0, -1.0),
    "LEFT_TO_RIGHT": (1.0, 0.0),
    "RIGHT_TO_LEFT": (-1.0, 0.0),
}
_RED_LIGHT_CROSSING_DIRECTIONS = {
    "NEGATIVE_TO_POSITIVE",
    "POSITIVE_TO_NEGATIVE",
}


def resolve_red_light_config(
    light_roi: Iterable[float],
    stop_line: Iterable[Iterable[float]],
    crossing_direction: str = "NEGATIVE_TO_POSITIVE",
) -> tuple[tuple[float, float, float, float], tuple[tuple[float, float], ...], str]:
    try:
        x1, y1, x2, y2 = (float(value) for value in light_roi)
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in stop_line
        )
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(
            "Traffic-light ROI and stop line must contain normalized coordinates."
        ) from error

    if (
        len(points) != 2
        or any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in (x1, y1, x2, y2, *[c for point in points for c in point])
        )
        or x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            "Traffic-light ROI must be a non-empty normalized rectangle and "
            "stop line must have two normalized endpoints."
        )
    if math.dist(points[0], points[1]) < 1e-5:
        raise ValueError("Stop line endpoints must be different.")
    direction = crossing_direction.strip().upper()
    if direction not in _RED_LIGHT_CROSSING_DIRECTIONS:
        raise ValueError(
            "Crossing direction must be NEGATIVE_TO_POSITIVE or "
            "POSITIVE_TO_NEGATIVE."
        )
    return (x1, y1, x2, y2), points, direction


class RedLightViolationDetector:
    def __init__(
        self,
        light_roi: Iterable[float],
        stop_line: Iterable[Iterable[float]],
        crossing_direction: str = "NEGATIVE_TO_POSITIVE",
        fps: float = 30.0,
    ) -> None:
        (
            self.light_roi,
            self.stop_line,
            self.crossing_direction,
        ) = resolve_red_light_config(light_roi, stop_line, crossing_direction)
        self.fps = max(float(fps), 1.0)
        self.signal_history: deque[str] = deque(
            maxlen=config.RED_LIGHT_STATE_HISTORY
        )
        self.signal_state = "UNKNOWN"
        self.track_states: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.signal_history.clear()
        self.signal_state = "UNKNOWN"
        self.track_states.clear()
        self.events.clear()

    def _signal_observation(self, frame: Any) -> str:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.light_roi
        left = max(0, min(width - 1, int(x1 * width)))
        top = max(0, min(height - 1, int(y1 * height)))
        right = max(left + 1, min(width, int(math.ceil(x2 * width))))
        bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        illuminated = (saturation >= 100) & (value >= 160)
        masks = {
            "RED": ((hue <= 10) | (hue >= 170)) & illuminated,
            "YELLOW": (hue >= 15) & (hue <= 40) & illuminated,
            "GREEN": (hue >= 40) & (hue <= 90) & illuminated,
        }
        counts = {
            color: int(np.count_nonzero(mask))
            for color, mask in masks.items()
        }
        threshold = max(
            config.RED_LIGHT_MIN_ACTIVE_PIXELS,
            int(roi.shape[0] * roi.shape[1] * config.RED_LIGHT_MIN_ACTIVE_ROI_RATIO),
        )
        winner, strongest = max(counts.items(), key=lambda item: item[1])
        runner_up = max(
            (count for color, count in counts.items() if color != winner),
            default=0,
        )
        if strongest < threshold or (
            runner_up
            and strongest < runner_up * config.RED_LIGHT_COLOR_DOMINANCE_RATIO
        ):
            return "UNKNOWN"
        return winner

    def _smooth_signal(self, observation: str) -> str:
        self.signal_history.append(observation)
        counts = {
            state: self.signal_history.count(state)
            for state in ("RED", "YELLOW", "GREEN")
        }
        winner = max(counts, key=counts.get)
        ordered = sorted(counts.values(), reverse=True)
        if (
            counts[winner] < config.RED_LIGHT_STATE_CONFIRMATION_FRAMES
            or len(ordered) > 1
            and ordered[0] == ordered[1]
            or self.signal_history[-1] != winner
        ):
            return "UNKNOWN"
        return winner

    def update(
        self,
        frame: Any,
        tracks: list[dict[str, Any]],
        frame_index: int,
    ) -> dict[str, Any]:
        raw_signal = self._signal_observation(frame)
        self.signal_state = self._smooth_signal(raw_signal)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.light_roi
        roi_box = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        cv2.rectangle(
            frame,
            roi_box[:2],
            roi_box[2:],
            config.RED_LIGHT_COLORS[self.signal_state],
            2,
        )
        cv2.putText(
            frame,
            f"Signal: {self.signal_state} (raw {raw_signal})",
            (roi_box[0], max(18, roi_box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.RED_LIGHT_COLORS[self.signal_state],
            2,
            cv2.LINE_AA,
        )

        (start_x, start_y), (end_x, end_y) = self.stop_line
        line_start = (round(start_x * width), round(start_y * height))
        line_end = (round(end_x * width), round(end_y * height))
        cv2.line(frame, line_start, line_end, (255, 220, 0), 3, cv2.LINE_AA)
        direction_sign = (
            1 if self.crossing_direction == "NEGATIVE_TO_POSITIVE" else -1
        )
        midpoint = (
            (line_start[0] + line_end[0]) // 2,
            (line_start[1] + line_end[1]) // 2,
        )
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        line_length = max(math.hypot(dx, dy), 1)
        arrow_end = (
            round(midpoint[0] + direction_sign * dy / line_length * 30),
            round(midpoint[1] - direction_sign * dx / line_length * 30),
        )
        cv2.arrowedLine(
            frame, midpoint, arrow_end, (255, 220, 0), 2, tipLength=0.3
        )

        deadband = min(height, width) * config.RED_LIGHT_CROSSING_DEADBAND_RATIO
        max_gap_frames = round(
            self.fps * config.RED_LIGHT_MAX_CROSSING_GAP_SECONDS
        )
        current_states = {}
        for record in tracks:
            vehicle_id = record["id"]
            box_x1, _, box_x2, box_y2 = record["box"]
            point = ((box_x1 + box_x2) / 2, box_y2)
            signed_distance = (
                dx * (point[1] - line_start[1])
                - dy * (point[0] - line_start[0])
            ) / line_length
            track = self.track_states.setdefault(
                vehicle_id,
                {
                    "vehicle_type": record["type"],
                    "stable_side": None,
                    "stable_point": point,
                    "stable_frame": frame_index,
                    "pending_side": None,
                    "pending_side_frames": 0,
                    "pending_signal": "UNKNOWN",
                    "last_seen_frame": frame_index,
                    "alerted": False,
                },
            )
            previous_side = track["stable_side"]
            new_side = (
                1 if signed_distance > deadband else
                -1 if signed_distance < -deadband else
                None
            )
            if new_side is None or new_side == previous_side:
                track["pending_side"] = None
                track["pending_side_frames"] = 0
            elif track["pending_side"] == new_side:
                track["pending_side_frames"] += 1
            else:
                track["pending_side"] = new_side
                track["pending_side_frames"] = 1
                track["pending_signal"] = self.signal_state

            if (
                new_side is not None
                and new_side != previous_side
                and track["pending_side_frames"]
                >= config.RED_LIGHT_CROSSING_CONFIRMATION_FRAMES
            ):
                gap = frame_index - track["stable_frame"]
                crossed_as_configured = (
                    previous_side is not None
                    and (
                        previous_side < new_side
                        if direction_sign > 0
                        else previous_side > new_side
                    )
                )
                movement = (
                    point[0] - track["stable_point"][0],
                    point[1] - track["stable_point"][1],
                )
                moved_through_line = (
                    movement[0] * movement[0] + movement[1] * movement[1]
                ) >= (deadband * 2) ** 2
                if (
                    crossed_as_configured
                    and gap <= max_gap_frames
                    and moved_through_line
                    and track["pending_signal"] == "RED"
                ):
                    track["alerted"] = True
                    self.events.append(
                        {
                            "vehicle_id": vehicle_id,
                            "vehicle_type": record["type"],
                            "frame": frame_index,
                            "signal": track["pending_signal"],
                        }
                    )
                track["stable_side"] = new_side
                track["stable_point"] = point
                track["stable_frame"] = frame_index
                track["pending_side"] = None
                track["pending_side_frames"] = 0
            track["last_seen_frame"] = frame_index
            track["vehicle_type"] = record["type"]
            current_states[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "state": "VIOLATION" if track["alerted"] else "CLEAR",
                "side": track["stable_side"],
            }

        stale_frames = max_gap_frames * 2
        for vehicle_id, track in list(self.track_states.items()):
            if frame_index - track["last_seen_frame"] > stale_frames:
                del self.track_states[vehicle_id]

        return {
            "signal_state": self.signal_state,
            "raw_signal_state": raw_signal,
            "vehicle_states": current_states,
        }


def resolve_lane_config(
    lane_direction: str = config.DEFAULT_LANE_DIRECTION,
    lane_polygon: Iterable[Iterable[float]] = config.DEFAULT_LANE_POLYGON,
) -> tuple[str, tuple[tuple[float, float], ...]]:
    direction = lane_direction.strip().upper()
    if direction not in _DIRECTION_VECTORS:
        raise ValueError(
            f"Unknown lane direction '{lane_direction}'. "
            f"Choose one of: {', '.join(config.LANE_DIRECTIONS)}."
        )

    try:
        points_list = []
        for point in lane_polygon:
            if len(point) != 2:
                raise ValueError
            points_list.append((float(point[0]), float(point[1])))
        points = tuple(points_list)
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(
            "Lane polygon must contain at least three normalized [x, y] points."
        ) from error
    if len(points) < 3 or any(
        not math.isfinite(value) or value < 0 or value > 1
        for point in points
        for value in point
    ):
        raise ValueError(
            "Lane polygon must contain at least three normalized [x, y] points "
            "between 0 and 1."
        )
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points, points[1:] + points[:1]
            )
        )
    )
    if area <= 1e-8:
        raise ValueError("Lane polygon points must enclose a non-zero area.")
    return direction, points


class WrongDirectionDetector:
    def __init__(
        self,
        lane_direction: str = config.DEFAULT_LANE_DIRECTION,
        lane_polygon: Iterable[Iterable[float]] = config.DEFAULT_LANE_POLYGON,
        fps: float = 30.0,
        debug: bool = config.DEBUG_WRONG_DIRECTION,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.debug = debug
        self.set_lane_config(lane_direction, lane_polygon)

    def set_lane_config(
        self,
        lane_direction: str,
        lane_polygon: Iterable[Iterable[float]],
    ) -> None:
        self.lane_direction, self.lane_polygon = resolve_lane_config(
            lane_direction, lane_polygon
        )
        self.expected_direction = _DIRECTION_VECTORS[self.lane_direction]
        self.reset()

    def reset(self) -> None:
        self.track_states: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def _direction_observation(
        self,
        track: dict[str, Any],
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> None:
        window_frames = max(2, round(self.fps * config.DIRECTION_WINDOW_SECONDS))
        target_frame = frame_index - window_frames
        history = track["lane_history"]
        old_point = next(
            (item for item in reversed(history) if item[0] <= target_frame),
            None,
        )
        if old_point is None:
            return

        old_frame, (old_x, old_y) = old_point
        new_frame, (new_x, new_y) = history[-1]
        if new_frame <= old_frame:
            return
        dx = new_x - old_x
        dy = new_y - old_y
        distance = math.hypot(dx, dy)
        min_displacement = min(frame_shape[0], frame_shape[1]) * (
            config.MIN_DIRECTION_DISPLACEMENT_RATIO
        )
        if distance < min_displacement:
            return

        movement = (dx / distance, dy / distance)
        expected_x, expected_y = self.expected_direction
        similarity = movement[0] * expected_x + movement[1] * expected_y
        along_lane = abs(dx * expected_x + dy * expected_y)
        across_lane = abs(dx * expected_y - dy * expected_x)
        track["movement"] = movement
        track["similarity"] = similarity

        if (
            abs(similarity) < config.DIRECTION_MIN_COSINE
            or along_lane < across_lane * config.DIRECTION_DOMINANCE_RATIO
        ):
            track["candidate"] = None
            track["candidate_frames"] = 0
            return

        candidate = (
            "CORRECT_DIRECTION" if similarity > 0 else "WRONG_DIRECTION"
        )
        if track["candidate"] == candidate:
            track["candidate_frames"] += 1
        else:
            track["candidate"] = candidate
            track["candidate_frames"] = 1

        if (
            track["candidate_frames"] >= config.DIRECTION_CONFIRMATION_FRAMES
            and track["state"] != "WRONG_DIRECTION"
        ):
            track["state"] = candidate
            if candidate == "WRONG_DIRECTION" and not track["alerted"]:
                track["alerted"] = True
                self.events.append(
                    {
                        "vehicle_id": track["vehicle_id"],
                        "vehicle_type": track["vehicle_type"],
                        "direction": self.lane_direction,
                        "confidence": abs(similarity),
                    }
                )

    def update(
        self,
        tracks: list[dict[str, Any]],
        frame: Any,
        frame_index: int,
    ) -> dict[str, dict[str, Any]]:
        height, width = frame.shape[:2]
        polygon = np.asarray(
            [
                (round(x * (width - 1)), round(y * (height - 1)))
                for x, y in self.lane_polygon
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [polygon], True, (0, 220, 255), 2)
        expected_x, expected_y = self.expected_direction
        arrow_end = (
            int(48 + expected_x * 36),
            int(52 + expected_y * 36),
        )
        cv2.arrowedLine(
            frame, (48, 52), arrow_end, (0, 220, 255), 2, tipLength=0.3
        )
        cv2.putText(
            frame,
            f"Lane direction: {self.lane_direction}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        states = {}
        for record in tracks:
            vehicle_id = record["id"]
            x1, _, x2, y2 = record["box"]
            point = ((x1 + x2) / 2, y2)
            track = self.track_states.setdefault(
                vehicle_id,
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_type": record["type"],
                    "history": deque(
                        maxlen=max(
                            config.DIRECTION_HISTORY_LENGTH,
                            round(self.fps * config.DIRECTION_WINDOW_SECONDS)
                            + config.DIRECTION_CONFIRMATION_FRAMES
                            + 2,
                        )
                    ),
                    "lane_history": deque(
                        maxlen=max(
                            config.DIRECTION_HISTORY_LENGTH,
                            round(self.fps * config.DIRECTION_WINDOW_SECONDS)
                            + config.DIRECTION_CONFIRMATION_FRAMES
                            + 2,
                        )
                    ),
                    "inside_frames": 0,
                    "in_lane": False,
                    "state": "UNKNOWN",
                    "candidate": None,
                    "candidate_frames": 0,
                    "movement": None,
                    "similarity": None,
                    "alerted": False,
                },
            )
            track["vehicle_type"] = record["type"]
            track["detection_confidence"] = record["confidence"]
            track["last_seen_frame"] = frame_index
            track["history"].append((frame_index, point))
            is_in_lane = cv2.pointPolygonTest(
                polygon, (float(point[0]), float(point[1])), False
            ) >= 0

            if is_in_lane:
                track["inside_frames"] += 1
                track["in_lane"] = (
                    track["inside_frames"] >= config.LANE_ENTRY_CONFIRMATION_FRAMES
                )
                if track["in_lane"]:
                    track["lane_history"].append((frame_index, point))
                    if track["state"] not in (
                        "CORRECT_DIRECTION",
                        "WRONG_DIRECTION",
                    ):
                        track["state"] = "IN_LANE"
                    if track["inside_frames"] >= config.LANE_ENTRY_CONFIRMATION_FRAMES:
                        self._direction_observation(track, frame_index, frame.shape)
            else:
                track["inside_frames"] = 0
                track["in_lane"] = False
                track["lane_history"].clear()
                track["candidate"] = None
                track["candidate_frames"] = 0
                if track["state"] not in (
                    "CORRECT_DIRECTION",
                    "WRONG_DIRECTION",
                ):
                    track["state"] = "NOT_RELEVANT"

            color = config.WRONG_DIRECTION_COLORS[track["state"]]
            cv2.putText(
                frame,
                f'{vehicle_id} {track["state"].replace("_DIRECTION", "")}',
                (x1, min(y2 + 20, height - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
            if self.debug:
                trajectory = [
                    (int(x), int(y)) for _, (x, y) in track["history"]
                ]
                if len(trajectory) > 1:
                    cv2.polylines(
                        frame,
                        [np.asarray(trajectory, dtype=np.int32)],
                        False,
                        (160, 160, 160),
                        2,
                    )
                movement = track["movement"]
                movement_text = (
                    "unconfirmed"
                    if movement is None
                    else f"({movement[0]:.2f}, {movement[1]:.2f})"
                )
                score_text = (
                    "--"
                    if track["similarity"] is None
                    else f"{track['similarity']:.2f}"
                )
                cv2.putText(
                    frame,
                    (
                        f"Lane: {'YES' if is_in_lane else 'NO'} "
                        f"Det: {record['confidence']:.2f} "
                        f"Move: {movement_text} "
                        f"Expected: ({expected_x:.1f}, {expected_y:.1f}) "
                        f"Score: {score_text}"
                    ),
                    (x1, max(16, y2 + 38)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            states[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "vehicle_type": track["vehicle_type"],
                "state": track["state"],
                "lane_member": is_in_lane,
                "detection_confidence": track["detection_confidence"],
                "confidence": (
                    abs(track["similarity"])
                    if track["similarity"] is not None
                    else None
                ),
            }

        max_age = round(self.fps * config.TRACK_STATE_TIMEOUT_SECONDS)
        for vehicle_id, track in list(self.track_states.items()):
            if frame_index - track.get("last_seen_frame", frame_index) > max_age:
                del self.track_states[vehicle_id]
        return states


def resolve_operations(selected: Iterable[str]) -> tuple[str, ...]:
    chosen = set(selected)
    unknown = chosen.difference(config.OPERATIONS)
    if unknown:
        raise ValueError(f"Unknown analysis operation(s): {', '.join(sorted(unknown))}")
    if not chosen:
        raise ValueError("Select at least one analysis operation.")

    pending = list(chosen)
    while pending:
        operation = pending.pop()
        for dependency in config.OPERATION_DEPENDENCIES[operation]:
            if dependency not in chosen:
                chosen.add(dependency)
                pending.append(dependency)
    return tuple(operation for operation in config.OPERATIONS if operation in chosen)


def validate_operation_media(media_suffix: str, operations: Iterable[str]) -> None:
    selected = set(operations)
    if (
        media_suffix.lower() in config.VIDEO_EXTENSIONS
        and selected.intersection(("plate", "ocr"))
        and not selected.intersection(config.VIOLATION_OPERATIONS)
    ):
        raise ValueError(
            "Number-plate detection and OCR without a violation operation "
            "are available for images only."
        )


@lru_cache(maxsize=3)
def _load_yolo(operation: str) -> YOLO:
    try:
        filename = _MODEL_FILENAMES[operation]
    except KeyError as error:
        raise ValueError(f"Unsupported YOLO operation: {operation}") from error

    model_path = config.MODEL_DIR / filename
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {operation} model: {model_path}. "
            "Add the trained model file to models/."
        )
    return YOLO(str(model_path))


@lru_cache(maxsize=1)
def _load_ocr() -> RapidOCR:
    return RapidOCR()


def _load_models(operations: tuple[str, ...]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for operation in ("vehicle", "helmet", "plate"):
        if operation in operations:
            models[operation] = _load_yolo(operation)
    if "ocr" in operations:
        models["ocr"] = _load_ocr()
    return models


def _model_name(names: Any, class_id: int) -> str:
    return str(names[class_id])


def _canonical_vehicle_name(name: str) -> str | None:
    return config.VEHICLE_CLASS_ALIASES.get(name.strip().lower())


def _clip_box(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in box)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _draw_box(
    frame: Any,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _extract_ocr_text(ocr_result: Any) -> tuple[str, float | None]:
    texts = getattr(ocr_result, "txts", None)
    scores = getattr(ocr_result, "scores", None)
    texts = [] if texts is None else texts
    scores = [] if scores is None else scores
    accepted: list[str] = []
    accepted_scores: list[float] = []

    for index, text in enumerate(texts):
        normalized = str(text).strip()
        if not normalized:
            continue
        score = float(scores[index]) if index < len(scores) else None
        if score is not None and score < config.OCR_MIN_CONFIDENCE:
            continue
        accepted.append(normalized)
        if score is not None:
            accepted_scores.append(score)

    if not accepted:
        return "", None
    confidence = sum(accepted_scores) / len(accepted_scores) if accepted_scores else None
    return " ".join(accepted), confidence


def _class_ids(names: Any, predicate: Any) -> list[int]:
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    return [class_id for class_id, name in entries if predicate(str(name))]


def _track_class_ids(vehicle_model: YOLO) -> list[int]:
    return _class_ids(
        vehicle_model.names,
        lambda name: _canonical_vehicle_name(name) in config.TRACKED_VEHICLES,
    )


def _plate_class_ids(plate_model: YOLO) -> list[int]:
    return _class_ids(plate_model.names, lambda name: "plate" in name.strip().lower())


def _reset_trackers(vehicle_model: YOLO) -> None:
    predictor = getattr(vehicle_model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def _process_frame(
    frame: Any,
    is_video: bool,
    models: dict[str, Any],
    operations: tuple[str, ...],
    plate_class_ids: list[int],
    vehicle_class_ids: list[int],
    recognized_plates: dict[str, dict[str, Any]],
    plate_boxes_by_vehicle: dict[str, tuple[int, int, int, int]],
    wrong_direction_detector: WrongDirectionDetector | None = None,
    red_light_detector: RedLightViolationDetector | None = None,
    frame_index: int = 0,
    plate_attempt_frames: dict[str, int] | None = None,
    ocr_attempt_frames: dict[str, int] | None = None,
) -> Any:
    plate_attempt_frames = plate_attempt_frames if plate_attempt_frames is not None else {}
    ocr_attempt_frames = ocr_attempt_frames if ocr_attempt_frames is not None else {}
    raw_frame = frame.copy()
    vehicle_model = models["vehicle"]
    results = vehicle_model.track(
        source=frame,
        persist=is_video,
        tracker=config.VEHICLE_TRACKER,
        classes=vehicle_class_ids,
        conf=config.VEHICLE_CONFIDENCE,
        iou=config.VEHICLE_IOU,
        imgsz=config.VEHICLE_IMAGE_SIZE,
        verbose=False,
    )
    height, width = frame.shape[:2]
    records: list[dict[str, Any]] = []
    boxes = (
        results[0].boxes
        if results and results[0].boxes is not None
        else []
    )
    for index, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        vehicle_type = _canonical_vehicle_name(_model_name(vehicle_model.names, class_id))
        if vehicle_type not in config.TRACKED_VEHICLES:
            continue
        track_id = int(box.id[0].item()) if box.id is not None else index + 1
        vehicle_id = f"V{track_id}"
        vehicle_box = _clip_box(box.xyxy[0].cpu().tolist(), width, height)
        confidence = float(box.conf[0].item())
        _draw_box(
            frame,
            vehicle_box,
            f"{vehicle_id} {vehicle_type} {confidence:.2f}",
            config.VEHICLE_COLORS.get(vehicle_type, (255, 255, 255)),
        )
        records.append(
            {
                "id": vehicle_id,
                "type": vehicle_type,
                "box": vehicle_box,
                "confidence": confidence,
            }
        )

    violation_labels: dict[str, list[str]] = {}
    if wrong_direction_detector is not None:
        direction_states = wrong_direction_detector.update(
            records, frame, frame_index
        )
        for vehicle_id, state in direction_states.items():
            if state["state"] == "WRONG_DIRECTION":
                violation_labels.setdefault(vehicle_id, []).append(
                    "WRONG DIRECTION"
                )
    if red_light_detector is not None:
        red_light_states = red_light_detector.update(frame, records, frame_index)
        for vehicle_id, state in red_light_states["vehicle_states"].items():
            if state["state"] == "VIOLATION":
                violation_labels.setdefault(vehicle_id, []).append(
                    "RED-LIGHT VIOLATION"
                )

    if "helmet" in operations:
        for vehicle_id in _detect_helmets_for_two_wheelers(
            frame, raw_frame, records, models["helmet"]
        ):
            violation_labels.setdefault(vehicle_id, []).append("NO HELMET")

    if "plate" in operations or "ocr" in operations:
        _detect_plates_for_vehicles(
            frame,
            raw_frame,
            models.get("plate"),
            models.get("ocr"),
            plate_class_ids,
            "plate" in operations,
            "ocr" in operations,
            recognized_plates,
            plate_boxes_by_vehicle,
            frame_index,
            plate_attempt_frames,
            ocr_attempt_frames,
            (
                [record for record in records if record["id"] in violation_labels]
                if set(operations).intersection(config.VIOLATION_OPERATIONS)
                else records
            ),
        )

    for record in records:
        labels = violation_labels.get(record["id"])
        if labels:
            _draw_box(
                frame,
                record["box"],
                f"{record['id']} {' · '.join(labels)}",
                config.RED_LIGHT_VIOLATION_COLOR,
            )
    return frame


def _detect_helmets_for_two_wheelers(
    frame: Any,
    raw_frame: Any,
    records: list[dict[str, Any]],
    helmet_model: YOLO,
) -> set[str]:
    two_wheelers = [r for r in records if r["type"] in config.HELMET_VEHICLES]
    if not two_wheelers:
        return set()

    result = helmet_model.predict(
        source=raw_frame,
        conf=config.HELMET_CONFIDENCE,
        imgsz=config.HELMET_IMAGE_SIZE,
        verbose=False,
    )[0]
    if result.boxes is None:
        return set()

    height, width = raw_frame.shape[:2]
    rider_regions = []
    for record in two_wheelers:
        x1, y1, x2, y2 = record["box"]
        vehicle_width = x2 - x1
        vehicle_height = y2 - y1
        rider_regions.append(
            (
                max(0, x1 - int(vehicle_width * config.HELMET_RIDER_REGION_SIDE_SCALE)),
                max(0, y1 - int(vehicle_height * config.HELMET_RIDER_REGION_TOP_SCALE)),
                min(width, x2 + int(vehicle_width * config.HELMET_RIDER_REGION_SIDE_SCALE)),
                y2,
                record["id"],
            )
        )

    violating_vehicle_ids = set()
    for box in result.boxes:
        helmet_box = _clip_box(
            box.xyxy[0].cpu().tolist(), width, height
        )
        center_x = (helmet_box[0] + helmet_box[2]) / 2
        center_y = (helmet_box[1] + helmet_box[3]) / 2
        matching_regions = [
            region
            for region in rider_regions
            if region[0] <= center_x <= region[2]
            and region[1] <= center_y <= region[3]
        ]
        if not matching_regions:
            continue

        region = min(
            matching_regions,
            key=lambda item: (
                (center_x - (item[0] + item[2]) / 2) / max(item[2] - item[0], 1)
            ) ** 2
            + ((center_y - (item[1] + item[3]) / 2) / max(item[3] - item[1], 1))
            ** 2,
        )
        label = _model_name(helmet_model.names, int(box.cls[0].item()))
        normalized_label = " ".join(
            label.lower().replace("_", " ").replace("-", " ").split()
        )
        is_violation = any(
            phrase in normalized_label
            for phrase in (
                "without helmet",
                "no helmet",
                "not wearing helmet",
                "helmetless",
            )
        )
        confidence = float(box.conf[0].item())
        _draw_box(
            frame,
            helmet_box,
            f"{region[4]} {label} {confidence:.2f}",
            (
                config.RED_LIGHT_VIOLATION_COLOR
                if is_violation
                else config.HELMET_COLOR
            ),
        )
        if is_violation:
            violating_vehicle_ids.add(region[4])
    return violating_vehicle_ids


def _detect_plates_for_vehicles(
    frame: Any,
    raw_frame: Any,
    plate_model: YOLO | None,
    ocr_engine: RapidOCR | None,
    plate_class_ids: list[int],
    use_plate: bool,
    use_ocr: bool,
    recognized_plates: dict[str, dict[str, Any]],
    plate_boxes_by_vehicle: dict[str, tuple[int, int, int, int]],
    frame_index: int,
    plate_attempt_frames: dict[str, int],
    ocr_attempt_frames: dict[str, int],
    target_records: list[dict[str, Any]],
) -> None:
    if use_plate and plate_model is None:
        raise ValueError("Number-plate detection was selected but its model is missing.")
    if use_ocr and ocr_engine is None:
        raise ValueError("OCR was selected but its engine is unavailable.")

    interval = config.PLATE_OCR_INTERVAL_FRAMES
    plate_due_records = [
        record
        for record in target_records
        if frame_index - plate_attempt_frames.get(record["id"], -interval)
        >= interval
    ]
    crops = []
    crop_records = []
    for record in (plate_due_records if use_plate else ()):
        plate_attempt_frames[record["id"]] = frame_index
        x1, y1, x2, y2 = record["box"]
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size:
            crops.append(crop)
            crop_records.append(record)

    batch_size = config.INFERENCE_BATCH_SIZE
    if use_plate and plate_model is not None:
        for start in range(0, len(crops), batch_size):
            batched_results = plate_model.predict(
                source=crops[start : start + batch_size],
                classes=plate_class_ids,
                conf=config.PLATE_CONFIDENCE,
                imgsz=config.PLATE_IMAGE_SIZE,
                verbose=False,
            )
            for record, result in zip(
                crop_records[start : start + batch_size], batched_results
            ):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                best_box = max(
                    result.boxes, key=lambda box: float(box.conf[0].item())
                )
                bx1, by1, bx2, by2 = _clip_box(
                    best_box.xyxy[0].cpu().tolist(),
                    record["box"][2] - record["box"][0],
                    record["box"][3] - record["box"][1],
                )
                plate_boxes_by_vehicle[record["id"]] = (bx1, by1, bx2, by2)

    for record in target_records:
        vehicle_id = record["id"]
        if use_plate:
            relative_box = plate_boxes_by_vehicle.get(vehicle_id)
            if relative_box is None:
                continue
            vx1, vy1, _, _ = record["box"]
            px1, py1, px2, py2 = relative_box
            target_box = _clip_box(
                (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2),
                frame.shape[1],
                frame.shape[0],
            )
        else:
            target_box = record["box"]

        label = f"{vehicle_id} plate" if use_plate else vehicle_id
        if (
            use_ocr
            and ocr_engine is not None
            and frame_index - ocr_attempt_frames.get(vehicle_id, -interval)
            >= interval
        ):
            x1, y1, x2, y2 = target_box
            crop = raw_frame[y1:y2, x1:x2]
            if crop.size:
                ocr_attempt_frames[vehicle_id] = frame_index
                text, confidence = _extract_ocr_text(ocr_engine(crop))
                previous = recognized_plates.get(vehicle_id)
                previous_confidence = (
                    previous.get("confidence") if previous is not None else None
                )
                if text and (
                    previous is None
                    or confidence is not None
                    and (
                        previous_confidence is None
                        or confidence > previous_confidence
                    )
                ):
                    recognized_plates[vehicle_id] = {
                        "vehicle_id": vehicle_id,
                        "text": text,
                        "confidence": confidence,
                    }
        if use_ocr and vehicle_id in recognized_plates:
            label = f"{vehicle_id} {recognized_plates[vehicle_id]['text']}"

        _draw_box(frame, target_box, label, config.PLATE_COLOR)


def _iter_video_frames(capture: Any):
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-reader") as reader:
        next_frame = reader.submit(capture.read)
        while True:
            ok, frame = next_frame.result()
            if not ok:
                break
            next_frame = reader.submit(capture.read)
            yield frame


def process_media(
    source_path: Path,
    analysis_dir: Path,
    selected_operations: Iterable[str] = config.OPERATIONS,
    lane_direction: str = config.DEFAULT_LANE_DIRECTION,
    lane_polygon: Iterable[Iterable[float]] = config.DEFAULT_LANE_POLYGON,
    light_roi: Iterable[float] | None = None,
    stop_line: Iterable[Iterable[float]] | None = None,
    crossing_direction: str = config.DEFAULT_RED_LIGHT_CROSSING_DIRECTION,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    operations = resolve_operations(selected_operations)
    lane_direction, lane_polygon = resolve_lane_config(
        lane_direction, lane_polygon
    )
    red_light_config = None
    if "red_light" in operations:
        if light_roi is None or stop_line is None:
            raise ValueError(
                "Configure the traffic-light region and stop line before analysis."
            )
        red_light_config = resolve_red_light_config(
            light_roi, stop_line, crossing_direction
        )
    suffix = source_path.suffix.lower()
    is_video = suffix in config.VIDEO_EXTENSIONS
    is_image = suffix in config.IMAGE_EXTENSIONS
    if not is_video and not is_image:
        raise ValueError(f"Unsupported media type: {suffix or '(no extension)'}")
    validate_operation_media(suffix, operations)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    if source_path.resolve().parent == analysis_dir.resolve():
        stored_source = source_path
    else:
        stored_source = analysis_dir / f"source{suffix}"
        shutil.copyfile(source_path, stored_source)
    output_path = analysis_dir / f"processed{suffix if is_image else '.webm'}"

    with _PROCESSING_LOCK:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "loading_models",
                    "progress": 0,
                    "current_frame": 0,
                    "total_frames": None,
                }
            )
        models = _load_models(operations)
        vehicle_model = models["vehicle"]
        vehicle_class_ids = _track_class_ids(vehicle_model)
        if not vehicle_class_ids:
            raise ValueError("Vehicle model has none of the configured vehicle classes.")

        plate_class_ids: list[int] = []
        if "plate" in operations:
            plate_class_ids = _plate_class_ids(models["plate"])
            if not plate_class_ids:
                raise ValueError("Plate model has no class with 'plate' in its name.")

        _reset_trackers(vehicle_model)
        recognized_plates: dict[str, dict[str, Any]] = {}
        plate_boxes_by_vehicle: dict[str, tuple[int, int, int, int]] = {}
        plate_attempt_frames: dict[str, int] = {}
        ocr_attempt_frames: dict[str, int] = {}
        wrong_direction_detector = None
        red_light_detector = None

        if is_image:
            if "wrong_direction" in operations:
                wrong_direction_detector = WrongDirectionDetector(
                    lane_direction,
                    lane_polygon,
                    fps=1.0,
                    debug=config.DEBUG_WRONG_DIRECTION,
                )
            if red_light_config is not None:
                red_light_detector = RedLightViolationDetector(
                    *red_light_config,
                    fps=1.0,
                )
            frame = cv2.imread(str(stored_source))
            if frame is None:
                raise ValueError(f"Could not read image: {source_path.name}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "processing",
                        "progress": 0,
                        "current_frame": 0,
                        "total_frames": 1,
                    }
                )
            processed = _process_frame(
                frame,
                False,
                models,
                operations,
                plate_class_ids,
                vehicle_class_ids,
                recognized_plates,
                plate_boxes_by_vehicle,
                wrong_direction_detector,
                red_light_detector,
                1,
                plate_attempt_frames,
                ocr_attempt_frames,
            )
            if not cv2.imwrite(str(output_path), processed):
                raise RuntimeError(f"Could not save processed image: {output_path}")
        else:
            capture = cv2.VideoCapture(str(stored_source))
            if not capture.isOpened():
                raise ValueError(f"Could not read video: {source_path.name}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30.0
            if "wrong_direction" in operations:
                wrong_direction_detector = WrongDirectionDetector(
                    lane_direction,
                    lane_polygon,
                    fps=fps,
                    debug=config.DEBUG_WRONG_DIRECTION,
                )
            if red_light_config is not None:
                red_light_detector = RedLightViolationDetector(
                    *red_light_config,
                    fps=fps,
                )
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "processing",
                        "progress": 0,
                        "current_frame": 0,
                        "total_frames": total_frames if total_frames > 0 else None,
                    }
                )
            if width <= 0 or height <= 0:
                capture.release()
                raise ValueError(f"Could not determine video dimensions: {source_path.name}")

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"VP80"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError("Could not create the processed video.")
            frames_written = 0
            try:
                for frame_index, frame in enumerate(
                    _iter_video_frames(capture), start=1
                ):
                    processed = _process_frame(
                        frame,
                        True,
                        models,
                        operations,
                        plate_class_ids,
                        vehicle_class_ids,
                        recognized_plates,
                        plate_boxes_by_vehicle,
                        wrong_direction_detector,
                        red_light_detector,
                        frame_index,
                        plate_attempt_frames,
                        ocr_attempt_frames,
                    )
                    writer.write(processed)
                    frames_written += 1
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "processing",
                                "progress": (
                                    min(
                                        99,
                                        round(frames_written * 100 / total_frames),
                                    )
                                    if total_frames > 0
                                    else 0
                                ),
                                "current_frame": frames_written,
                                "total_frames": (
                                    total_frames if total_frames > 0 else None
                                ),
                            }
                        )
            finally:
                capture.release()
                writer.release()
            if frames_written == 0:
                output_path.unlink(missing_ok=True)
                raise RuntimeError("The input video contains no readable frames.")
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "finalizing",
                        "progress": 99,
                        "current_frame": frames_written,
                        "total_frames": total_frames if total_frames > 0 else None,
                    }
                )

    if progress_callback is not None and is_image:
        progress_callback(
            {
                "phase": "finalizing",
                "progress": 99,
                "current_frame": 1,
                "total_frames": 1,
            }
        )
    analysis_id = analysis_dir.name
    return {
        "analysis_id": analysis_id,
        "media_type": "video" if is_video else "image",
        "result_url": f"/results/{analysis_id}/{output_path.name}",
        "operations": list(operations),
        "ocr_results": list(recognized_plates.values()),
        "lane_direction": (
            lane_direction if wrong_direction_detector is not None else None
        ),
        "wrong_direction_events": (
            wrong_direction_detector.events
            if wrong_direction_detector is not None
            else []
        ),
        "wrong_direction_states": (
            [
                {
                    "vehicle_id": track["vehicle_id"],
                    "vehicle_type": track["vehicle_type"],
                    "state": track["state"],
                    "lane_member": track["in_lane"],
                    "confidence": (
                        abs(track["similarity"])
                        if track["similarity"] is not None
                        else None
                    ),
                }
                for track in wrong_direction_detector.track_states.values()
            ]
            if wrong_direction_detector is not None
            else []
        ),
        "red_light_signal_state": (
            red_light_detector.signal_state
            if red_light_detector is not None
            else None
        ),
        "red_light_events": (
            red_light_detector.events
            if red_light_detector is not None
            else []
        ),
        "red_light_configuration": (
            {
                "light_roi": red_light_detector.light_roi,
                "stop_line": red_light_detector.stop_line,
                "crossing_direction": red_light_detector.crossing_direction,
            }
            if red_light_detector is not None
            else None
        ),
    }


def convert_video_for_browser(source_path: Path, output_path: Path) -> Path:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read saved video: {source_path.name}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine video dimensions: {source_path.name}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create a browser-compatible video.")

    try:
        frames_written = 0
        try:
            for frame in _iter_video_frames(capture):
                writer.write(frame)
                frames_written += 1
        finally:
            capture.release()
            writer.release()
        if frames_written == 0:
            raise RuntimeError(
                f"Saved video contains no readable frames: {source_path.name}"
            )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def new_analysis_id() -> str:
    return uuid.uuid4().hex
