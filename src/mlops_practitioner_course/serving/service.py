import bentoml
from mlops_practitioner_course.api import app 
from mlops_practitioner_course.serving.model_service import ModelService

@bentoml.service
@bentoml.asgi_app(app, path="/")
class Gateway:
    model_service = bentoml.depends(ModelService)
