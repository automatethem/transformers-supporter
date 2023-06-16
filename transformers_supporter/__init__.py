from . import models
from . import pipelines

from .models.ann import configuration_ann
from .models.ann import feature_extraction_ann
from .models.ann import modeling_ann

from .models.cnn import configuration_cnn
from .models.cnn import modeling_cnn
from .models.cnn import image_processing_cnn

from .models.custom_bert import configuration_custom_bert
from .models.custom_bert import modeling_custom_bert

from .models.custom_wav2vec2 import feature_extraction_custom_wav2vec2

from .models.embedded_1dcnnimport import configuration_embedded_1dcnn
from .models.embedded_1dcnnimport import modeling_embedded_1dcnn


def register_auto():
    configuration_ann.register_auto()
    modeling_ann.register_auto()
    feature_extraction_ann.register_auto()

    configuration_cnn.register_auto()
    modeling_cnn.register_auto()
    image_processing_cnn.register_auto()

    configuration_custom_bert.register_auto()
    modeling_custom_bert.register_auto()

    feature_extraction_custom_wav2vec2.register_auto()

    configuration_embedded_1dcnn.register_auto()
    modeling_embedded_1dcnn.register_auto()
