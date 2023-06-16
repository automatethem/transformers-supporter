from . import models
from . import pipelines

from transformers_supporter.models.cnn.configuration_cnn import CnnConfig
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification

from transformers_supporter.models.ann.configuration_ann import AnnConfig
from transformers_supporter.models.ann.modeling_ann import CnnForImageClassification

from .models.ann import modeling_ann
from .models.cnn import configuration_cnn
from .models.cnn import modeling_cnn
def register_auto():
    modeling_ann.register_auto()
    modeling_cnn.register_auto()
    configuration_cnn.register_auto()
