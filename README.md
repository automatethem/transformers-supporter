# transformers-supporter

https://pypi.org/project/transformers-supporter
```
pip install transformers-supporter
```

## Supported models

```
import transformers_supporter
transformers_supporter.register_auto()
```

### Ann

```
#transformers_supporter.models.AnnConfig
#transformers_supporter.models.AnnForTabularRegression
#transformers_supporter.models.TabularFeatureExtractor
AutoConfig
AnnForTabularRegression
AutoFeatureExtractor
pipeline(task="tabular-regression") (transformers_supporter.pipelines.TabularRegressionPipeline)
```

```
#transformers_supporter.models.AnnConfig
#transformers_supporter.models.AnnForTabularClassification
#transformers_supporter.models.TabularFeatureExtractor
AutoConfig
AnnForTabularClassification
AutoFeatureExtractor
pipeline(task="tabular-classification") (transformers_supporter.pipelines.TabularClassificationPipeline)
```


```
transformers_supporter.models.AnnForTabularBinaryClassification
```

### Cnn

```
#transformers_supporter.models.CnnConfig
#transformers_supporter.models.CnnForImageClassification
#AutoImageProcessor
AutoConfig
AutoModelForImageClassification
AutoImageProcessor
pipeline(task="image-classification")
```

```
#transformers_supporter.models.CnnConfig
#transformers_supporter.models.CnnForImageClassification
#transformers_supporter.models.GrayscaleImageProcessor
AutoConfig
AutoModelForImageClassification
AutoImageProcessor
pipeline(task="image-classification")
```

```
transformers_supporter.models.CnnForKeyPointDetection
```

#### Faster Rcnn

```
#transformers_supporter.models.FasterRcnnConfig
#transformers_supporter.models.FasterRcnnForObjectDetection
#transformers_supporter.models.FasterRcnnImageProcessor
AutoConfig
AutoModelForObjectDetection
AutoImageProcessor
pipeline(task="object-detection")
```

### Rnn

```
transformers_supporter.models.RnnConfig
transformers_supporter.models.RnnForAudioClassification
transformers_supporter.models.RnnForTimeSeriesRegression
```

#### Embedded Rnn

```
#EmbeddedRnnConfig
#EmbeddedRnnForSequenceClassification
#TorchtextFeatureExtractor
AutoConfig
AutoModelForSequenceClassification
AutoFeatureExtractor
```

#### Pretrained Embedded Rnn

```
#PretrainedEmbeddedRnnConfig
#PretrainedEmbeddedRnnForSequenceClassification
#TorchtextFeatureExtractor
AutoConfig
AutoModelForSequenceClassification
AutoFeatureExtractor
```

#### Embedded 1dcnn

```
#transformers_supporter.models.Embedded1dcnnConfig
#transformers_supporter.models.Embedded1dcnnForSequenceClassification
#transformers_helper.models.TorchtextFeatureExtractor
AutoConfig
AutoModelForForSequenceClassification
AutoFeatureExtractor
pipeline("text-classification")
```

```
transformers_supporter.models.EmbeddedRnnForFixedLengthTranslation
```

### Bert

#### Custom Bert

```
transformers_supporter.models.CustomBertConfig
transformers_supporter.models.CustomBertForSequenceClassification
```

#### Custom Wav2Vec2

```
transformers_supporter.models.CustomWav2Vec2FeatureExtractor
```

## Supported pipelines

```
import transformers_supporter
transformers_supporter.register_pipeline()
```



### CustomImageClassificationPipeline

```
transformers_supporter.pipelines.CustomImageClassificationPipeline
```

### FixedLengthTranslationPipeline

```
transformers_supporter.pipelines.FixedLengthTranslationPipeline
```


### TabularBinaryClassificationPipeline

```
transformers_supporter.pipelines.TabularBinaryClassificationPipeline
```

