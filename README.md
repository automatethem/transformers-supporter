# transformers-supporter

https://pypi.org/project/transformers-supporter
```
pip install transformers-supporter
```

## Supported models

```
import transformers_supporter
transformers_supporter.register_auto()
transformers_supporter.register_pipeline()
```

### Ann


#### tabular-regression

```
#transformers_supporter.models.AnnConfig
#transformers_supporter.models.AnnForTabularRegression
#transformers_supporter.models.TabularFeatureExtractor
AutoConfig
AnnForTabularRegression
AutoFeatureExtractor
pipeline(task="tabular-regression") (transformers_supporter.pipelines.TabularRegressionPipeline)
```

#### tabular-classification

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

#### image-classification

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

#### KeyPointDetection
```
transformers_supporter.models.CnnForKeyPointDetection
```

#### object-detection

```
#transformers_supporter.models.FasterRcnnConfig
#transformers_supporter.models.FasterRcnnForObjectDetection
#transformers_supporter.models.FasterRcnnImageProcessor
AutoConfig
AutoModelForObjectDetection
AutoImageProcessor
pipeline(task="object-detection")
```

#### text-classification

```
#transformers_supporter.models.Embedded1dcnnConfig
#transformers_supporter.models.Embedded1dcnnForSequenceClassification
#transformers_helper.models.TorchtextFeatureExtractor
AutoConfig
AutoModelForForSequenceClassification
AutoFeatureExtractor
pipeline("text-classification")
```

### Rnn

#### audio-classification

```
#transformers_supporter.models.RnnConfig
#transformers_supporter.models.RnnForAudioClassification
#AutoFeatureExtractor
AutoConfig
AutoModelForAudioClassification
AutoFeatureExtractor
pipeline(task="audio-classification")
```

#### TimeSeriesRegression

```
transformers_supporter.models.RnnForTimeSeriesRegression
```

#### text-classification

```
#EmbeddedRnnConfig
#EmbeddedRnnForSequenceClassification
#TorchtextFeatureExtractor
AutoConfig
AutoModelForSequenceClassification
AutoFeatureExtractor
```

```
#PretrainedEmbeddedRnnConfig
#PretrainedEmbeddedRnnForSequenceClassification
#TorchtextFeatureExtractor
AutoConfig
AutoModelForSequenceClassification
AutoFeatureExtractor
```


```
transformers_supporter.models.EmbeddedRnnForFixedLengthTranslation
```

### Bert

#### text-classification

```
transformers_supporter.models.CustomBertConfig
transformers_supporter.models.CustomBertForSequenceClassification
```

#### Custom Wav2Vec2

```
transformers_supporter.models.CustomWav2Vec2FeatureExtractor
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

