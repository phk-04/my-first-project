from sklearn.datasets import load_iris#꽃 데이터가져옴 
from sklearn.tree import DecisionTreeClassifier#트리형식머신러닝?가져옴
from sklearn.model_selection import train_test_split#테스트용 학습용 나누는함수 가져옴 
from sklearn.metrics import accuracy_score# 정확도 알려주는?거 가져옴 
import pandas as pd #판다스

# 데이터 불러오기
iris = load_iris()

X = iris.data
y = iris.target

# train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.9,
    random_state=11
)

# 모델 생성
model = DecisionTreeClassifier(random_state=11)

# 학습
model.fit(X_train, y_train)

# 예측
pred = model.predict(X_test)

# 평가
acc = accuracy_score(y_test, pred)

print("정확도:", acc)
