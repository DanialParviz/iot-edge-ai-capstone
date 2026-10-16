import math

import pytest

from edge.vision.face import UNKNOWN, best_face, cosine_similarity, crop, identify
from edge.vision.records import detection_records, face_record


def test_detection_records_drop_weak_detections():
    recs = detection_records([("person", 0.82), ("cup", 0.3), ("bird", 0.5)], min_score=0.5)
    assert [r["object"] for r in recs] == ["person", "bird"]
    assert recs[0]["confidence"] == 0.82 and recs[0]["sensor"] == "vision"


def test_face_record_shape():
    rec = face_record("Person1", 0.71234)
    assert rec["sensor"] == "face" and rec["identity"] == "Person1" and rec["similarity"] == 0.712


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # zero vector: no crash


def test_cosine_similarity_rejects_size_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_identify_known_and_unknown():
    refs = {"Person1": [1.0, 0.0, 0.0]}
    name, sim = identify([0.9, 0.1, 0.0], refs, threshold=0.5)
    assert name == "Person1" and sim > 0.9
    name, sim = identify([0.0, 1.0, 0.0], refs, threshold=0.5)
    assert name == UNKNOWN and sim == pytest.approx(0.0)


def test_identify_threshold_is_inclusive_and_picks_best_reference():
    refs = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    assert identify([0.0, 2.0], refs)[0] == "B"
    # exactly on the threshold counts as a match (cos 60 deg = 0.5)
    on_edge = [math.cos(math.radians(60)), math.sin(math.radians(60))]
    assert identify(on_edge, {"A": [1.0, 0.0]}, threshold=cosine_similarity(on_edge, [1.0, 0.0]))[0] == "A"


def test_identify_with_no_references_is_unknown():
    assert identify([1.0, 0.0], {})[0] == UNKNOWN


def test_best_face_picks_largest():
    assert best_face([(0, 0, 10, 10), (5, 5, 40, 30)]) == (5, 5, 40, 30)
    assert best_face([]) is None


class Frame:
    shape = (100, 200, 3)

    def __getitem__(self, key):
        self.last = key
        return key


def test_crop_clamps_to_frame():
    frame = Frame()
    ys, xs = crop(frame, (190, 90, 20, 20), margin=0.5)
    assert xs.stop == 200 and ys.stop == 100 and xs.start >= 0 and ys.start >= 0
