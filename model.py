import numpy as np

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor 

# ランダムフォレスト
def randomForest_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse


# ランダムフォレスト（作品推薦用）
def randomForest_model_recommendation(X_train, X_valid, y_train, y_valid, X_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred