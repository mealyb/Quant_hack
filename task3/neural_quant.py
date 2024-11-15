import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, log_loss

# Загрузка и подготовка данных
file_path = 'task-3-dataset.csv'
data = pd.read_csv(file_path)

# Преобразование меток в бинарные значения (1 для положительных, 0 для отрицательных)
data['разметка'] = data['разметка'].apply(lambda x: 1 if x == '+' else 0)

# Выделение признаков и меток
reviews = data['отзывы'].values
labels = data['разметка'].values

# Векторизация отзывов
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews).toarray()
y = np.array(labels)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Настройка квантовой модели
feature_dim = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim)
ansatz = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)
sampler = StatevectorSampler()

# Кастомный оптимизатор для отслеживания потерь
class LossTrackingCOBYLA(COBYLA):
    def __init__(self, maxiter=100):
        super().__init__(maxiter=maxiter)
        self.loss_history = []

    def step(self, loss):
        self.loss_history.append(loss)
        return super().step(loss)

# Используем кастомный оптимизатор
optimizer = LossTrackingCOBYLA(maxiter=20)  # Количество итераций, которое хотим контролировать

# Инициализация VQC с использованием кастомного оптимизатора
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler)

# Обучение модели
vqc.fit(X_train, y_train)

# Кривая потерь на тренировочных данных
train_losses = optimizer.loss_history

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
