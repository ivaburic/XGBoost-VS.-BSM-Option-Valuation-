import numpy as np
import xgboost as xgb


class XGBoostModel:
    def __init__(self,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=200,
                 objective="reg:squarederror",
                 subsample=0.8,
                 colsample_bytree=0.8,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 min_child_weight=5,
                 gamma=0.0):

        self.model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            objective=objective,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=42,
            tree_method="hist",
            eval_metric="rmse",
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class EvaluationMetrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        mse = EvaluationMetrics.mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

    @staticmethod
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def evaluate_model(y_true, y_pred):
        mse = EvaluationMetrics.mean_squared_error(y_true, y_pred)
        mae = EvaluationMetrics.mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = EvaluationMetrics.r_squared(y_true, y_pred)
        return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R^2": r2}
