## Ann

### tabular-regression

son-height-tabular-regression-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/son-height-tabular-regression-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.ann.configuration_ann import AnnConfig
from transformers_supporter.models.ann.modeling_ann import AnnForTabularRegression
from transformers_supporter.models.ann.feature_extraction_ann import TabularFeatureExtractor

feature_extractor = TabularFeatureExtractor(
    continuous_columns=['Father'],
    categorical_columns=[],
    labels_column='Son',
    labels_shape='one',
    labels_type='float'
)
feature_extractor.train(train_df)

config = AnnConfig(
    in_features=1
)
model = AnnForTabularRegression(config)
model = model.to(device)

model_path = '/Users/automatethem/models/son-height-tabular-regression-model-transformers-custom-supporter'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.ann.modeling_ann import AnnForTabularRegression
from transformers_supporter.models.ann.feature_extraction_ann import TabularFeatureExtractor
from transformers_supporter.pipelines.tabular_regression import TabularRegressionPipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/son-height-tabular-regression-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/son-height-tabular-regression-model-transformers-custom-supporter'
model = AnnForTabularRegression.from_pretrained(model_path)
feature_extractor = TabularFeatureExtractor.from_pretrained(model_path)
pl = TabularRegressionPipeline(model=model, feature_extractor=feature_extractor, device=device)
```

```
print(dir(TabularRegressionPipeline)) #['__abstractmethods__', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_ensure_tensor_on_device', '_forward', '_sanitize_parameters', 'check_model_type', 'default_input_names', 'device_placement', 'ensure_tensor_on_device', 'forward', 'get_inference_context', 'get_iterator', 'iterate', 'postprocess', 'predict', 'preprocess', 'run_multi', 'run_single', 'save_pretrained', 'transform']
```

### tabular-classification

iris-tabular-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/iris-tabular-classification-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.ann.configuration_ann import AnnConfig
from transformers_supporter.models.ann.modeling_ann import AnnForTabularClassification
from transformers_supporter.models.ann.feature_extraction_ann import TabularFeatureExtractor

feature_extractor = TabularFeatureExtractor(
    continuous_columns=['Father'],
    categorical_columns=[],
    labels_column='Son',
    labels_shape='one',
    labels_type='float'
)
feature_extractor.train(train_df)

config = AnnConfig(
    in_features=4,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = AnnForTabularClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/iris-tabular-classification-model-transformers-custom-supporter'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.ann.modeling_ann import AnnForTabularRegression
from transformers_supporter.models.ann.feature_extraction_ann import TabularFeatureExtractor
from transformers_supporter.pipelines.tabular_regression import TabularClassificationPipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/iris-tabular-classification-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/iris-tabular-classification-model-transformers-custom-supporter'
model = AnnForTabularClassification.from_pretrained(model_path)
feature_extractor = TabularFeatureExtractor.from_pretrained(model_path)
pl = TabularClassificationPipeline(model=model, feature_extractor=feature_extractor, device=device)
```

