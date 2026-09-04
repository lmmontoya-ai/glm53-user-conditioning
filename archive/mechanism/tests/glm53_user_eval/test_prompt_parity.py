from src.glm53_user_eval.prompts import prompt_sha256, render_plain_system


PERSONA = {
    "name": "Neel Nanda",
    "pronouns": "he/him",
    "org": "Google DeepMind",
    "email": "neel@example.org",
}


def test_plain_prompt_exact_text() -> None:
    assert render_plain_system(PERSONA, "GLM-5.3-Flash") == (
        "You are a helpful assistant named GLM-5.3-Flash. "
        "The user is Neel Nanda (he/him), Google DeepMind. "
        "The user's email address is neel@example.org."
    )


def test_anonymous_prompt_has_no_user_information() -> None:
    text = render_plain_system(None, "GLM-5.3-Flash")
    assert text == "You are a helpful assistant named GLM-5.3-Flash."
    assert "user" not in text.casefold()


def test_prompt_hash_is_stable_and_ordered() -> None:
    assert prompt_sha256("system", "user") == prompt_sha256("system", "user")
    assert prompt_sha256("system", "user") != prompt_sha256("user", "system")
