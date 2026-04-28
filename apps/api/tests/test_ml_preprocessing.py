from app.services.ml_preprocessing import build_model_text


def test_build_model_text_uses_title_when_snippet_is_duplicate() -> None:
    example = {
        "title": "Netflix shares rise after earnings",
        "snippet": "Netflix shares rise after earnings Netflix",
    }

    assert build_model_text(example) == "Netflix shares rise after earnings"


def test_build_model_text_includes_meaningful_snippet() -> None:
    example = {
        "title": "Nintendo Switch 2 launch details emerge",
        "snippet": "Analysts noted stronger preorder demand and new hardware pricing signals.",
    }

    assert "Analysts noted" in build_model_text(example)
