from . import models
from . import pipelines

from .models.ann import modeling_ann
from .models.cnn import configuration_cnn
from .models.cnn import modeling_cnn
def register_auto():
    modeling_ann.register_auto()
    modeling_cnn.register_auto()
    configuration_cnn.register_auto()
