"""Extension point for the research method, whose design is still pending."""

from src.models.base import BaseForecaster


class ProposedForecaster(BaseForecaster):
    name = "proposed"

    def __init__(self, **params):
        self.params = params

    def fit(self, batch):
        raise NotImplementedError("The proposed research method has not been defined. Implement this class before including it in experiments.")

    def predict(self, inputs):
        raise NotImplementedError("The proposed research method has not been defined.")
