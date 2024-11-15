import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import SPSA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Загрузка данных
file_path = 'task-3-dataset.csv'
data = pd.read_csv(file_path)
data['разметка'] = data['разметка'].apply(lambda x: 1 if x == '+' else 0)

# Преобразование данных
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['отзывы']).toarray()
y = np.array(data['разметка'])

# Снижение размерности с использованием PCA
pca = PCA(n_components=3)  # Уменьшили размерность до 3 признаков
X_reduced = pca.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Настройка квантовой модели с уменьшенной размерностью
feature_dim = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim)
ansatz = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=5)  # Увеличили сложность анзаца
sampler = StatevectorSampler()

# Массив для хранения потерь
train_losses = []

# Callback функция для отслеживания потерь
def callback(objective_weights, objective_value):
    train_losses.append(objective_value)
    print(f"Loss = {objective_value}")

# Настройка оптимизатора
optimizer = SPSA(maxiter=100)  # Используем SPSA и увеличили maxiter

# Инициализация VQC с использованием callback
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler, callback=callback)

# Обучение модели
vqc.fit(X_train, y_train)

# Оценка модели на тестовых данных
y_test_pred = vqc.predict(X_test)
print(classification_report(y_test, y_test_pred))

# Визуализация кривой функции потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
