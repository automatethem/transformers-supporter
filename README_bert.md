# Bert

## text-classification

### english

imdb-movie-review-text-classification-model-transformers-custom-bert-supporter  
https://github.com/automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-bert-supporter

```
import torch
from transformers_supporter.models.custom_bert.configuration_custom_bert import CustomBertConfig
from transformers_supporter.models.custom_bert.modeling_custom_bert import CustomBertForSequenceClassification
from transformers import AutoTokenizer

model_path = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)

config = CustomBertConfig(
    bert_model_path=model_path, 
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = CustomBertForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-bert-supporter'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.custom_bert.modeling_custom_bert import CustomBertForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/imdb-movie-review-text-classification-model-transformers-custom-bert-supporter'
model_path = 'automatethem-back-model/imdb-movie-review-text-classification-model-transformers-custom-bert-supporter'
model = CustomBertForSequenceClassification.from_pretrained(model_path)
model_path = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)
```

### korean

naver-movie-review-text-classification-model-transformers-custom-bert-supporter  
https://github.com/automatethem-back-model/naver-movie-review-text-classification-model-transformers-custom-bert-supporter

```
import torch
from transformers_supporter.models.custom_bert.configuration_custom_bert import CustomBertConfig
from transformers_supporter.models.custom_bert.modeling_custom_bert import CustomBertForSequenceClassification
from transformers import AutoTokenizer

model_path = 'snunlp/KR-FinBert-SC'
tokenizer = AutoTokenizer.from_pretrained(model_path)

config = CustomBertConfig(
    bert_model_path=model_path, 
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model = CustomBertForSequenceClassification(config)
model = model.to(device)

model_path = '/Users/automatethem/models/naver-movie-review-text-classification-model-transformers-custom-bert-supporter'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

```
import torch
from transformers_supporter.models.custom_bert.modeling_custom_bert import CustomBertForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

#model_path = '/Users/automatethem/models/naver-movie-review-text-classification-model-transformers-custom-bert-supporter'
model_path = 'automatethem-back-model/naver-movie-review-text-classification-model-transformers-custom-bert-supporter'
model = CustomBertForSequenceClassification.from_pretrained(model_path)
model_path = 'snunlp/KR-FinBert-SC'
tokenizer = AutoTokenizer.from_pretrained(model_path)
pl = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)
```
