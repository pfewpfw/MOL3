import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ======================================================
# Этап 1. Подготовка данных
# ======================================================

# Чтение датасета
df = pd.read_csv('D:/для работ/MMOlab/titanic_data.csv')

# Отображение всех столбцов
pd.options.display.max_columns = None

print("Первые 10 строк датасета:")
print(df.head(10))

# Подсчёт пропущенных значений по столбцам
print("\nКоличество пропущенных значений по столбцам:")
print(df.isnull().sum())

# Заполнение пропусков:
# - Для числовых столбцов — медианой
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# - Для категориальных столбцов — модой
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())

# Нормализация числовых признаков в диапазоне [0, 1]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Преобразование категориальных признаков в dummy-переменные (drop_first=True для избежания мультиколлинеарности)
df = pd.get_dummies(df, drop_first=True)

print("\nДатасет после нормализации и преобразования категориальных данных:")
print(df.head())

# ======================================================
# Этап 2. Задача регрессии (прогноз 'Fare')
# ======================================================

# Проверка наличия столбца 'Fare'
if 'Fare' not in df.columns:
    raise ValueError("В датасете нет столбца 'Fare'! Выберите другой непрерывный признак для регрессии.")

# Отделяем целевой признак от остальных
X_reg = df.drop('Fare', axis=1)
y_reg = df['Fare']

# Разделение данных на обучающую и тестовую выборки (80%/20%)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Создаем и обучаем модель линейной регрессии
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Предсказание на тестовой выборке
y_pred_reg = reg_model.predict(X_test_reg)

# Рассчитываем метрики для регрессии
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== Задача регрессии (прогноз 'Fare') ===")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# ======================================================
# Этап 3. Задача классификации с оценкой через ROC-кривую (прогноз 'Survived')
# ======================================================

# Проверка наличия столбца 'Survived'
if 'Survived' not in df.columns:
    raise ValueError("В датасете нет столбца 'Survived'! Проверьте целевой признак для классификации.")

# Отделяем признаки от целевой переменной
X_clf = df.drop('Survived', axis=1)
y_clf = df['Survived']

# Разделяем данные на обучающую и тестовую выборки (80%/20%)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Создаем и обучаем модель логистической регрессии
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf, y_train_clf)

# Получаем предсказанные вероятности для положительного класса (обычно класс 1)
y_prob = clf_model.predict_proba(X_test_clf)[:, 1]

# Вычисляем ROC-кривую и площадь под кривой (AUC)
fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob)
roc_auc = roc_auc_score(y_test_clf, y_prob)

print("\n=== Задача классификации (прогноз 'Survived') ===")
print("ROC AUC:", roc_auc)

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % roc_auc, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая для модели логистической регрессии')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
