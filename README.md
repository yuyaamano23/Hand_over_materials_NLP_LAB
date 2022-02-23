# このリポジトリは引き継ぎ用資料です

[天野の卒業論文:GoogleDrive](https://drive.google.com/drive/folders/1LA1KtRjWnPh6JUiVz5q9mpHXj9EgHxzS?usp=sharing)

# Sentence-BERT について

## 計算手順

### 1. アウトソーシングで集計したデータをから Sentence-BERT の J-E スコアを算出する

大学の研究室 PC を用いて実行しているため松井さんに聞く必要がある。

- アウトソーシングで集計したデータ:[base.csv](https://github.com/yuyaamano23/Hand_over_materials_NLP_LAB/blob/main/sentencebert/data/base.csv)

- Sentence-BERT の J-E スコア:[アウトソーシング検証用データ.txt(2 行目)](https://github.com/yuyaamano23/Hand_over_materials_NLP_LAB/blob/main/sentencebert/%E3%82%A2%E3%82%A6%E3%83%88%E3%82%BD%E3%83%BC%E3%82%B7%E3%83%B3%E3%82%B0%E6%A4%9C%E8%A8%BC%E7%94%A8%E3%83%87%E3%83%BC%E3%82%BF.txt#L2)

- 使用したモデルは「[paraphrase-multilingual-mpnet-base-v2](https://www.sbert.net/docs/pretrained_models.html#:~:text=paraphrase%2D-,multilingual,-%2Dmpnet%2Dbase%2Dv2)」

> 元論文:[Making Monolingual Sentence Embeddings Multilingual using
> Knowledge Distillation](https://arxiv.org/pdf/2004.09813.pdf)

### 2.

# ROC,AUC 計算コード

自分は[GoogleColab](https://colab.research.google.com/)で計算していました。

`test_y = []`には[各採点基準 ①②③④ の 01 の値](https://github.com/yuyaamano23/Hand_over_materials_NLP_LAB/blob/main/sentencebert/%E3%82%A2%E3%82%A6%E3%83%88%E3%82%BD%E3%83%BC%E3%82%B7%E3%83%B3%E3%82%B0%E6%A4%9C%E8%A8%BC%E7%94%A8%E3%83%87%E3%83%BC%E3%82%BF.txt#L8)を、 `predict_y = []`には[J-Esocore](https://github.com/yuyaamano23/Hand_over_materials_NLP_LAB/blob/main/sentencebert/%E3%82%A2%E3%82%A6%E3%83%88%E3%82%BD%E3%83%BC%E3%82%B7%E3%83%B3%E3%82%B0%E6%A4%9C%E8%A8%BC%E7%94%A8%E3%83%87%E3%83%BC%E3%82%BF.txt#L2)を入れる。

```python
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# test_y = [0,1,0,...]
test_y = []
# predict_y = [0.909771681 ,0.891080797 ,0.926733315,...]
predict_y = []


# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y)

auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='ROC curve (area = %.4f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
```
