from traitlets import HasTraits, Bool
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler as SkLearnStandardScaler
from logging import getLogger

log = getLogger(__name__)

class StandardScaler(BaseEstimator, HasTraits):
    with_mean = Bool(default_value=True)
    with_std = Bool(default_value=True)
    copy = Bool(default_value=True)
    per_channel = Bool(default_value=True)

    def __init__(self, copy: bool = copy.default_value, with_mean: bool = with_mean.default_value,
                 with_std: bool = with_std.default_value, per_channel : bool = per_channel.default_value):
        """

        :param copy: If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.
        :param with_mean: If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
        :param with_std: If True, scale the data to unit variance (or equivalently, unit standard deviation).
        :param per_channel: If True scale per channel, otherwise scale per sample
        """
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.per_channel = per_channel
        self.scalers = {}

    def partial_fit(self, X, y=None, sample_weight=None):
        if self.per_channel:
            num_channels = X.shape[-1]
            for channel_id in range(num_channels):
                if channel_id in self.scalers:
                    scaler = self.scalers[channel_id]
                else:
                    self.scalers[channel_id] = SkLearnStandardScaler(with_std=self.with_std, with_mean=self.with_mean, copy=self.copy)
                    scaler = self.scalers[channel_id]
                data = X[..., channel_id]
                data = data.reshape(-1, data.shape[-1])
                scaler.partial_fit(data, None, sample_weight=sample_weight)
        else:
            if not self.scalers:
                self.scalers = SkLearnStandardScaler(with_std=self.with_std, with_mean=self.with_mean, copy=self.copy)
            data = X.reshape(-1, X.shape[-1])
            self.scalers.partial_fit(data, None, sample_weight=sample_weight)
