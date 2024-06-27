import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv('knn_accuracy.csv')

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(data['K'], data['Accuracy'], marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.grid(True)
plt.savefig('knn_accuracy.png')
plt.show()