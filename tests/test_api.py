from mlops_practitioner_course.api import app


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict(client):
    response = client.post("/predict", json={"text": "تغريدة جميلة"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    assert response.status_code == 200
    assert len(response.json()) == 3




def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
