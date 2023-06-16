from . import models
from . import pipelines

from transformers_supporter.models.cnn.configuration_cnn import CnnConfig
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification
def register_auto():
    AutoConfig.register("cnn", CnnConfig)
    AutoModelForImageClassification.register(CnnConfig, CnnForImageClassification)
  
