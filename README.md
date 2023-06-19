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
AutoConfig (transformers_supporter.models.AnnConfig)
AnnForTabularRegression (transformers_supporter.models.AnnForTabularRegression)
AutoFeatureExtractor (transformers_supporter.models.TabularFeatureExtractor)
pipeline(task="tabular-regression") (transformers_supporter.pipelines.TabularRegressionPipeline)
```

#### tabular-classification

```
AutoConfig (transformers_supporter.models.AnnConfig)
AnnForTabularClassification (transformers_supporter.models.AnnForTabularClassification)
AutoFeatureExtractor (transformers_supporter.models.TabularFeatureExtractor)
pipeline(task="tabular-classification") (transformers_supporter.pipelines.TabularClassificationPipeline)
```


```
transformers_supporter.models.AnnForTabularBinaryClassification
```

### Cnn

#### image-classification

```
AutoConfig (transformers_supporter.models.CnnConfig)
AutoModelForImageClassification (transformers_supporter.models.CnnForImageClassification)
AutoImageProcessor (AutoImageProcessor)
pipeline(task="image-classification")
```

```
AutoConfig (transformers_supporter.models.CnnConfig)
AutoModelForImageClassification (transformers_supporter.models.CnnForImageClassification)
AutoImageProcessor (transformers_supporter.models.GrayscaleImageProcessor)
pipeline(task="image-classification")
```

#### object-detection

```
AutoConfig (transformers_supporter.models.FasterRcnnConfig)
AutoModelForObjectDetection (transformers_supporter.models.FasterRcnnForObjectDetection)
AutoImageProcessor (transformers_supporter.models.FasterRcnnImageProcessor)
pipeline(task="object-detection")
```

#### KeyPointDetection
```
transformers_supporter.models.CnnForKeyPointDetection
```

#### text-classification

```
AutoConfig (transformers_supporter.models.Embedded1dcnnConfig)
AutoModelForForSequenceClassification (transformers_supporter.models.Embedded1dcnnForSequenceClassification)
AutoFeatureExtractor (transformers_helper.models.TorchtextFeatureExtractor)
pipeline("text-classification")
```

### Rnn

#### text-classification

```
AutoConfig (EmbeddedRnnConfig)
AutoModelForSequenceClassification (EmbeddedRnnForSequenceClassification)
AutoFeatureExtractor (TorchtextFeatureExtractor)
```

```
AutoConfig (PretrainedEmbeddedRnnConfig)
AutoModelForSequenceClassification (PretrainedEmbeddedRnnForSequenceClassification)
AutoFeatureExtractor (TorchtextFeatureExtractor)
```

#### audio-classification

```
AutoConfig (transformers_supporter.models.RnnConfig)
AutoModelForAudioClassification (transformers_supporter.models.RnnForAudioClassification)
AutoFeatureExtractor (AutoFeatureExtractor)
pipeline(task="audio-classification")
```

#### TimeSeriesRegression

```
transformers_supporter.models.RnnForTimeSeriesRegression
```



```
transformers_supporter.models.EmbeddedRnnForFixedLengthTranslation
```

### Bert

#### text-classification

```
AutoConfig (transformers_supporter.models.CustomBertConfig)
AutoModelForSequenceClassification (transformers_supporter.models.CustomBertForSequenceClassification)
AutoTokenizer
pipeline(task="text-classification")
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

