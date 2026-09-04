from types import SimpleNamespace

import pytest
import torch

from src.glm53_user_eval.mhc import resolve_glm53_text_layers, select_prompt_vectors
from src.glm53_user_eval.runtime import prompt_final_indices, validate_glm53_config


def test_config_contract_matches_official_shape() -> None:
    assert validate_glm53_config(
        {"text_config": {"num_hidden_layers": 45, "hidden_size": 4096, "hc_mult": 4}}
    ) == {"text_layers": 45, "hidden_size": 4096, "hc_mult": 4}


def test_config_contract_fails_on_layer_mismatch() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_glm53_config(
            {"text_config": {"num_hidden_layers": 44, "hidden_size": 4096, "hc_mult": 4}}
        )


def test_layer_resolver_primary_path() -> None:
    layers = [object(), object()]
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    _, observed = resolve_glm53_text_layers(model)
    assert observed == layers


def test_layer_resolver_fails_closed() -> None:
    with pytest.raises(RuntimeError):
        resolve_glm53_text_layers(SimpleNamespace())


def test_prompt_final_indices_support_left_and_right_padding() -> None:
    assert prompt_final_indices(torch.tensor([[0, 1, 1], [1, 1, 0]])) == [2, 1]


def test_prompt_vector_selection_moves_only_small_tensor_to_cpu() -> None:
    streams = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    selected = select_prompt_vectors(streams, [2, 1])
    assert selected.shape == (2, 5)
    assert selected.device.type == "cpu"
