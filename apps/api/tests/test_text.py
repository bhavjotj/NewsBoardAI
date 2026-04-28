from app.utils.text import clean_snippet


def test_clean_snippet_removes_repeated_source_name() -> None:
    snippet = "Nintendo Switch 2 gets new launch details - Example News"

    cleaned = clean_snippet(snippet, source="Example News")

    assert cleaned == "Nintendo Switch 2 gets new launch details"


def test_clean_snippet_shortens_long_text() -> None:
    snippet = " ".join(["pricing"] * 80)

    cleaned = clean_snippet(snippet, max_length=80)

    assert len(cleaned) <= 80
    assert cleaned.endswith(".")
