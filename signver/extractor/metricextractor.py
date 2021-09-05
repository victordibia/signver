from signver.extractor import BaseExtractor


class MetricExtractor(BaseExtractor):
    def __init__(self, model_type="metric", batch_size=64):
        BaseExtractor.__init__(self,
                               model_type=model_type, batch_size=64)
