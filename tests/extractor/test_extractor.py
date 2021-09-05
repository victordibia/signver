from logging import log

from tensorflow.python.ops.variables import model_variables
from signver.extractor import MetricExtractor


def test_extractor_load():
    model_path = "models/extractor/metric"
    extractor = MetricExtractor()
    extractor.load(model_path)
    assert extractor.model is not None
