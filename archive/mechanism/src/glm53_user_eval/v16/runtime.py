"""V16 wrapper around the exact B300 runtime that passed V11."""

from src.glm53_user_eval.v11.runtime import (
    DownstreamForward,
    LoadedV11GLM53,
    SourceFeatures,
    pool_layer_streams,
)


class LoadedV16GLM53(LoadedV11GLM53):
    """The unchanged exact-checkpoint runtime under a V16 name."""


__all__ = ["DownstreamForward", "LoadedV16GLM53", "SourceFeatures", "pool_layer_streams"]
