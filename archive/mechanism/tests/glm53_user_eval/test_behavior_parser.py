import pytest

from src.glm53_user_eval.behavior import (
    condition_missingness_spread,
    parse_binary_judge,
    parse_direct_binary_response,
    parse_confidence_judge,
)


def test_direct_binary_parser_is_strict() -> None:
    assert parse_direct_binary_response("No.") == "no"
    assert parse_direct_binary_response(" YES ") == "yes"
    assert parse_direct_binary_response("I think no") is None


BINARY_CASES = [
    ("ANSWER = YES", "yes"),
    ("answer=yes", "yes"),
    ("Reason\nANSWER = NO", "no"),
    ("ANSWER = MAYBE", None),
    ("YES", None),
    ("", None),
    ("ANSWER = YESISH", None),
    ("X ANSWER = no X", "no"),
    ("ANSWER=YES\nconfidence=80", "yes"),
    ("REFUSED", None),
]


@pytest.mark.parametrize(("text", "expected"), BINARY_CASES)
def test_binary_parser_fixtures(text: str, expected: str | None) -> None:
    assert parse_binary_judge(text) == expected


CONFIDENCE_CASES = [
    ("ANSWER = 0%", 0.0, False, True),
    ("ANSWER = 1%", 1.0, False, True),
    ("ANSWER = 5%", 5.0, False, True),
    ("ANSWER = 10%", 10.0, False, True),
    ("ANSWER = 25%", 25.0, False, True),
    ("ANSWER = 50%", 50.0, False, True),
    ("ANSWER = 75%", 75.0, False, True),
    ("ANSWER = 99%", 99.0, False, True),
    ("ANSWER = 100%", 100.0, False, True),
    ("answer=87%", 87.0, False, True),
    ("Reason\nANSWER = 42%", 42.0, False, True),
    ("ANSWER = 20-40%", 30.0, False, True),
    ("ANSWER = 20–40%", 30.0, False, True),
    ("ANSWER = 0-100%", 50.0, False, True),
    ("ANSWER = 90-100%", 95.0, False, True),
    ("ANSWER = 40-20%", None, False, False),
    ("REFUSED", None, True, False),
    ("The assistant REFUSED", None, True, False),
    ("ANSWER = REFUSED", None, True, False),
    ("", None, False, False),
    ("87%", None, False, False),
    ("ANSWER = YES", None, False, False),
    ("ANSWER = -1%", None, False, False),
    ("ANSWER = 101%", None, False, False),
    ("ANSWER = ten percent", None, False, False),
    ("ANSWER: 50%", None, False, False),
    ("ANSWER = 7.5%", 7.0, False, True),
    ("prefix ANSWER = 33% suffix", 33.0, False, True),
    ("ANSWER= 66 %", 66.0, False, True),
    ("ANSWER = 09%", 9.0, False, True),
    ("ANSWER = 00%", 0.0, False, True),
    ("ANSWER = 100.0%", 100.0, False, True),
    ("ANSWER = 25-25%", 25.0, False, True),
    ("ANSWER = 1-2%", 1.5, False, True),
    ("NO ANSWER", None, False, False),
    ("ANSWER =", None, False, False),
    ("REFUSED ANSWER = 50%", None, True, False),
    ("ANSWER = 50 percent", None, False, False),
    ("ANSWER == 50%", None, False, False),
    ("ANSWER = 5%%", 5.0, False, True),
]


@pytest.mark.parametrize(("text", "value", "refused", "valid"), CONFIDENCE_CASES)
def test_confidence_parser_fixtures(
    text: str, value: float | None, refused: bool, valid: bool
) -> None:
    parsed = parse_confidence_judge(text)
    assert parsed.value == value
    assert parsed.refused is refused
    assert parsed.valid is valid


def test_missingness_spread_is_in_percentage_points() -> None:
    assert condition_missingness_spread(
        {"a": [True] * 10, "b": [True] * 8 + [False] * 2}
    ) == pytest.approx(20.0)


def test_missingness_requires_nonempty_cells() -> None:
    with pytest.raises(ValueError):
        condition_missingness_spread({"a": []})
