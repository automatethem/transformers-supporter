## Rnn

### text-classification

imdb-movie-review-text-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.embedded_rnn.configuration_embedded_rnn import EmbeddedRnnConfig
from transformers_supporter.models.embedded_rnn.modeling_embedded_rnn import EmbeddedRnnForSequenceClassification
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

config = EmbeddedRnnConfig(
    vocab_size=vocab_size,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = EmbeddedRnnForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-supporter'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.embedded_rnn.modeling_embedded_rnn import EmbeddedRnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-supporter'
model = EmbeddedRnnForSequenceClassification.from_pretrained(model_path)
feature_extractor = TorchtextFeatureExtractor.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=feature_extractor, device=device)
```

imdb-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-supporte  
https://github.com/automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-supporte

```
import torch
from transformers_supporter.models.pretrained_embedded_rnn.configuration_pretrained_embedded_rnn import PretrainedEmbeddedRnnConfig
from transformers_supporter.models.pretrained_embedded_rnn.modeling_pretrained_embedded_rnn import PretrainedEmbeddedRnnForSequenceClassification
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

config = PretrainedEmbeddedRnnConfig(
    vocab_size=vocab_size,
    id_to_token=feature_extractor.get_id_to_token(), #
    language='en', #
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = PretrainedEmbeddedRnnForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-supporte'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.pretrained_embedded_rnn.modeling_pretrained_embedded_rnn import PretrainedEmbeddedRnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-supporte'
model_path = 'automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-supporte'
model = PretrainedEmbeddedRnnForSequenceClassification.from_pretrained(model_path)
feature_extractor = TorchtextFeatureExtractor.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=feature_extractor, device=device)
```

naver-movie-review-text-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/naver-movie-review-text-classification-model-transformers-custom-supporter

naver-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-support  
https://github.com/automatethem-back-model/naver-movie-review-text-classification-model-transformers-custom-pretrained-embedded-rnn-support

car-name-text-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/car-name-text-classification-model-transformers-custom-supporter

```
import torch
from transformers_supporter.models.embedded_rnn.configuration_embedded_rnn import EmbeddedRnnConfig
from transformers_supporter.models.embedded_rnn.modeling_embedded_rnn import EmbeddedRnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor

feature_extractor = TorchtextFeatureExtractor(
    token_type='char',
    language='ko',
    min_freq=1
)
def text_iterator():
    for example in train_dataset:
        text = example['text']
        #ext = pytorch_supporter.utils.clean_korean(text)
        yield text
feature_extractor.train_from_iterator(text_iterator())

config = EmbeddedRnnConfig(
    vocab_size=vocab_size,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = EmbeddedRnnForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/car-name-text-classification-model-transformers-custom-supporter'
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.embedded_rnn.modeling_embedded_rnn import EmbeddedRnnForSequenceClassification
from transformers_supporter.models.embedded_rnn.feature_extraction_embedded_rnn import TorchtextFeatureExtractor
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/car-name-text-classification-model-transformers-custom-supporter'
model_path = 'automatethem-back-model/car-name-text-classification-model-transformers-custom-supporter'
model = EmbeddedRnnForSequenceClassification.from_pretrained(model_path)
feature_extractor = TorchtextFeatureExtractor.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=feature_extractor, device=device)
```

### audio-classification

speech-command-audio-classification-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/speech-command-audio-classification-model-transformers-custom-supporter

### translation

char-fixed-length-translation-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/char-fixed-length-translation-model-transformers-custom-supporter  

word-fixed-length-translation-model-transformers-custom-supporter  
https://github.com/automatethem-back-model/word-fixed-length-translation-model-transformers-custom-supporter