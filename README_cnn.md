## Cnn

### image-classification

rock-paper-scissors-image-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/rock-paper-scissors-image-classification-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.cnn.configuration_cnn import CnnConfig
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification
from transformers import AutoImageProcessor

model_path = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(model_path)

config = CnnConfig(
    in_channels=3,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = CnnForImageClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/rock-paper-scissors-image-classification-model-transformers-custom-supporter'
model.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification
from transformers import AutoImageProcessor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/rock-paper-scissors-image-classification-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/rock-paper-scissors-image-classification-model-transformers-custom-supporter'
model = CnnForImageClassification.from_pretrained(model_path)
model_path = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(model_path) 
pl = pipeline(task='image-classification', model=model, image_processor=image_processor, device=device)
```

mnist-hand-written-digit-image-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/mnist-hand-written-digit-image-classification-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.cnn.configuration_cnn import CnnConfig
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification
from transformers_supporter.models.cnn.image_processing_cnn import GrayscaleImageProcessor

image_processor = GrayscaleImageProcessor()

config = CnnConfig(
    in_channels=1,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = CnnForImageClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/mnist-hand-written-digit-image-classification-model-transformers-custom-supporter'
model.save_pretrained(model_path)
image_processor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.cnn.modeling_cnn import CnnForImageClassification
from transformers_supporter.models.cnn.image_processing_cnn import GrayscaleImageProcessor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/rock-paper-scissors-image-classification-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/mnist-hand-written-digit-image-classification-model-transformers-custom-supporter'
model = CnnForImageClassification.from_pretrained(model_path)
image_processor = GrayscaleImageProcessor.from_pretrained(model_path) 
pl = pipeline(task='image-classification', model=model, image_processor=image_processor, device=device)
```

### object-detection

illustration-object-detection-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/illustration-object-detection-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.faster_rcnn.configuration_faster_rcnn import FasterRcnnConfig
from transformers_supporter.models.faster_rcnn.modeling_faster_rcnn import FasterRcnnForObjectDetection
from transformers_supporter.models.faster_rcnn.image_processing_faster_rcnn import FasterRcnnImageProcessor

image_processor = FasterRcnnImageProcessor()

config = FasterRcnnConfig(
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = FasterRcnnForObjectDetection(config)
model = model.to(device)

model_path = '/Users/automatethem/models/illustration-object-detection-model-transformers-custom-supporter'
model.save_pretrained(model_path)
image_processor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.faster_rcnn.modeling_faster_rcnn import FasterRcnnForObjectDetection
from transformers_supporter.models.faster_rcnn.image_processing_faster_rcnn import FasterRcnnImageProcessor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/illustration-object-detection-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/illustration-object-detection-model-transformers-custom-supporter'
model = FasterRcnnForObjectDetection.from_pretrained(model_path)
image_processor = FasterRcnnImageProcessor.from_pretrained(model_path)
pl = pipeline(task='object-detection', model=model, image_processor=image_processor, device=device)
```

wheat-head-object-detection-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/wheat-head-object-detection-model-transformers-custom-supporter

### text-classification

imdb-movie-review-text-classification-model-transformers-custom-1dcnn-supporter  
https://github.com/automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-1dcnn-supporter

```
import torch
from transformers_supporter.models.embedded_1dcnn.configuration_embedded_1dcnn import Embedded1dcnnConfig
from transformers_supporter.models.embedded_1dcnn.modeling_embedded_1dcnn import Embedded1dcnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor

feature_extractor = TorchtextFeatureExtractor(
    token_type='word',
    language='en',
    min_freq=2
)
def text_iterator():
    for example in train_dataset:
        text = example['text']
        text = pytorch_supporter.utils.clean_english(text)
        yield text
feature_extractor.train_from_iterator(text_iterator())

config = Embedded1dcnnConfig(
    vocab_size=vocab_size,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = Embedded1dcnnForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/imdb-movie-review-text-classification-model-transformers-custom-1dcnn-supporter'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.embedded_1dcnn.modeling_embedded_1dcnn import Embedded1dcnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/imdb-movie-review-text-classification-model-transformers-custom-1dcnn-supporter'
model_path = 'automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-1dcnn-supporter'
model = Embedded1dcnnForSequenceClassification.from_pretrained(model_path)
tokenizer = TorchtextFeatureExtractor.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)
```
