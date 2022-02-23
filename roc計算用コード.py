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