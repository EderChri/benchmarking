from merlion.models.forecast.base import ForecasterBase, ForecasterConfig

class LSTMForecasterConfig(ForecasterConfig):
    def __init__(self, hidden_size=64, num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

class LSTMForecaster(ForecasterBase):
    config_class = LSTMForecasterConfig
    
    def train(self, train_data, train_config=None):
        pass
    
    def forecast(self, time_stamps, time_series_prev=None):
        pass
