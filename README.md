# このリポジトリは引き継ぎ用資料です

[天野の卒業論文:GoogleDrive](https://drive.google.com/drive/folders/1LA1KtRjWnPh6JUiVz5q9mpHXj9EgHxzS?usp=sharing)

# Sentence-BERT について

Sentence-BERT は大学の研究室 PC を用いて実行している。

(松井さんに聞く必要がある。)

使用したモデルは「[paraphrase-multilingual-mpnet-base-v2](https://www.sbert.net/docs/pretrained_models.html#:~:text=paraphrase%2D-,multilingual,-%2Dmpnet%2Dbase%2Dv2)」

元論文:[Making Monolingual Sentence Embeddings Multilingual using
Knowledge Distillation](https://arxiv.org/pdf/2004.09813.pdf)

# ROC,AUC 計算コード

自分は[GoogleColab](https://colab.research.google.com/)で実装していました。

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
