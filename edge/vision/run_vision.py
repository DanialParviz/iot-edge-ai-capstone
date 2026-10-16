"""Camera loop: capture -> detect objects (+ faces) -> store -> publish.

    python -m edge.vision.run_vision --model Models/efficientdet_lite0_uint8.tflite
    python -m edge.vision.run_vision --model ... --faces --yunet YUNET.onnx --reference Person1.pt
    python -m edge.vision.run_vision ... --show      # live window (needs the Pi desktop)

Headless by default, so it also runs over Raspberry Pi Connect's remote shell
(an OpenCV window cannot open there). The LED controller is a separate process;
this program only publishes.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from edge.config import load_config
from edge.storage import store_for
from edge.vision.records import FACE_TOPIC, detection_records, face_record

log = logging.getLogger("vision")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CE 4201 vision node")
    parser.add_argument("--model", required=True, help="EfficientDet-Lite0 .tflite file")
    parser.add_argument("--faces", action="store_true", help="also run face recognition")
    parser.add_argument("--yunet", help="YuNet .onnx file (with --faces)")
    parser.add_argument("--reference", help="reference embedding .pt for 'Person1' (with --faces)")
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--mqtt", action="store_true")
    parser.add_argument("--aws", action="store_true")
    parser.add_argument("--show", action="store_true", help="open a live window")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    args = parser.parse_args(argv)
    if args.faces and not (args.yunet and args.reference):
        parser.error("--faces needs --yunet and --reference")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import cv2
    from picamera2 import Picamera2

    from edge.vision.detector import MediapipeDetector

    config = load_config()
    store = store_for("vision", config.data_dir)
    publisher = None
    if args.mqtt or args.aws:
        from edge.cloud import build_publisher

        publisher = build_publisher(config, "pi5-vision", args.mqtt, args.aws)

    detector = MediapipeDetector(args.model, args.min_score)
    face_det = embedder = None
    references = {}
    if args.faces:
        from edge.vision.face import EdgeFaceEmbedder, YuNetDetector, load_reference

        face_det, embedder = YuNetDetector(args.yunet), EdgeFaceEmbedder()
        references = {"Person1": load_reference(args.reference)}

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    camera.start()

    def emit(topic: str, record: dict) -> None:
        store.insert(record)
        if publisher is not None:
            try:
                publisher.publish(topic, record)
            except Exception:
                log.exception("publish failed")

    try:
        frame_no = 0
        while args.frames == 0 or frame_no < args.frames:
            # Picamera2's "RGB888" arrays are laid out B,G,R in memory.
            bgr = camera.capture_array()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            for record in detection_records(detector.detect(rgb), args.min_score):
                emit(config.topics.vision, record)

            if face_det is not None:
                from edge.vision.face import best_face, crop, identify

                box = best_face(face_det.detect(bgr))
                if box is not None:
                    name, sim = identify(
                        embedder.embed(crop(rgb, box)), references, args.face_threshold)
                    emit(FACE_TOPIC, face_record(name, sim))

            if args.show:
                cv2.imshow("vision", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()
        store.close()
        if publisher is not None:
            publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
