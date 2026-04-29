from pathlib import Path

from app.services.torch_topic_service import TorchTopicService


def test_torch_topic_service_handles_missing_model_files(tmp_path: Path) -> None:
    service = TorchTopicService(model_dir=tmp_path / "missing")

    prediction = service.predict("Apple AI update", "New software tools launch.")

    assert service.available is False
    assert prediction.label is None
    assert prediction.available is False
    assert prediction.error
