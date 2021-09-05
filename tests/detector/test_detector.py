from signver.detector import Detector


def test_localizer_load():
    model_path = "models/detector/small"
    detector = Detector()
    detector.load(model_path)

    print(detector, detector.model_load_time)
    assert detector.model is not None
