from . import configuration_embedded_1dcnn
from . import modeling_embedded_1dcnn

def register():
    configuration_embedded_1dcnn.register_auto()
    modeling_embedded_1dcnn.register_auto()
