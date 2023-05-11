import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mypackage.models.model import MyModelClass
from mypackage.utils.data_processing import preprocess_data
from mypackage.config import DATA_PATH, TEST_DATA_PATH, MODEL_NAME, OUTPUT_DIR
from mypackage.utils.model_evaluation import evaluate_model
import joblib
from keras.models import load_model
from mypackage.utils.utils import load_data


def split_data(X, y):
    """
    データをトレーニングセットとテストセットに分割します。

    Parameters:
        X (array-like): 特徴量の配列または行列。
        y (array-like): ラベルの配列または行列。

    Returns:
        X_train (array-like): トレーニングセットの特徴量の配列または行列。
        X_test (array-like): テストセットの特徴量の配列または行列。
        y_train (array-like): トレーニングセットのラベルの配列または行列。
        y_test (array-like): テストセットのラベルの配列または行列。

    """
    # データの分割（トレーニングセットとテストセット）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    与えられたトレーニングデータを用いてモデルを学習します。

    Parameters:
        X_train (array-like): トレーニングセットの特徴量の配列または行列。
        y_train (array-like): トレーニングセットのラベルの配列または行列。

    Returns:
        model (MyModelClass): 学習済みのモデルオブジェクト。

    """
    # モデルの初期化
    model = MyModelClass()

    # モデルの学習
    model.train(X_train, y_train)

    return model


def hyperparameter_tuning(model, X_train, y_train):
    """
    ハイパーパラメータのチューニングを行い、最適なモデルとハイパーパラメータを返します。

    Parameters:
        model (MyModelClass): ベースモデルオブジェクト。
        X_train (array-like): トレーニングセットの特徴量の配列または行列。
        y_train (array-like): トレーニングセットのラベルの配列または行列。

    Returns:
        best_model (MyModelClass): 最適なモデルオブジェクト。
        best_params (dict): 最適なハイパーパラメータの辞書。

    """
    # ハイパーパラメータの設定
    parameters = {
        'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 128)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd']
    }

    # ハイパーパラメータのランダムサンプリング
    randomized_search = RandomizedSearchCV(model.model, parameters, cv=3)
    randomized_search.fit(X_train, y_train)

    # 最適なモデルとハイパーパラメータを取得
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_

    return best_model, best_params


def save_model(model):
    """
    モデルを保存します。

    Parameters:
        model (MyModelClass): 保存するモデルオブジェクト。

    Returns:
        None

    """
    # モデルの保存パス
    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)

    # モデルの保存
    model.model.save(model_path)


def save_pipeline(pipeline):
    """
    パイプラインを保存します。

    Parameters:
        pipeline (Pipeline): 保存するパイプラインオブジェクト。

    Returns:
        None

    """
    # パイプラインの保存パス
    preprocessing_pipeline_path = os.path.join(OUTPUT_DIR, 'preprocessing_pipeline.pkl')

    # パイプラインの保存
    joblib.dump(pipeline, preprocessing_pipeline_path)


def load_saved_model():
    """
    保存されたモデルを読み込みます。

    Returns:
        loaded_model (keras.models.Model): 読み込まれたモデルオブジェクト。

    """
    # モデルの読み込みパス
    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)

    # モデルの読み込み
    loaded_model = load_model(model_path)

    return loaded_model


def load_saved_pipeline():
    """
    保存されたパイプラインを読み込みます。

    Returns:
        loaded_pipeline (sklearn.pipeline.Pipeline): 読み込まれたパイプラインオブジェクト。

    """
    # パイプラインの読み込みパス
    preprocessing_pipeline_path = os.path.join(OUTPUT_DIR, 'preprocessing_pipeline.pkl')

    # パイプラインの読み込み
    loaded_pipeline = joblib.load(preprocessing_pipeline_path)

    return loaded_pipeline


def reuse_model(model, pipeline, X_test, y_test):
    """
    保存されたモデルとパイプラインを使用してテストデータを再利用します。

    Args:
        model (keras.models.Model): 読み込まれたモデルオブジェクト。
        pipeline (sklearn.pipeline.Pipeline): 読み込まれたパイプラインオブジェクト。
        X_test (numpy.ndarray): テストデータの特徴量行列。
        y_test (numpy.ndarray): テストデータのラベルベクトル。

    Returns:
        None

    """
    # データの前処理
    X_test = pipeline.transform(X_test)

    # モデルの再利用
    prediction = model.predict(X_test)
    print("Prediction:", prediction)


def evaluate_saved_model(model, X_test, y_test):
    """
    保存されたモデルを使用してテストデータを評価します。

    Args:
        model (keras.models.Model): 読み込まれたモデルオブジェクト。
        X_test (numpy.ndarray): テストデータの特徴量行列。
        y_test (numpy.ndarray): テストデータのラベルベクトル。

    Returns:
        None

    """
    # モデルの評価
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def main():
    """
    メインの実行関数です。データの読み込み、前処理、モデルの学習、ハイパーパラメータチューニング、
    モデルの保存と読み込み、モデルの再利用、保存されたモデルの評価の一連の処理を行います。

    Args:
        None

    Returns:
        None

    """
    # データの読み込み
    data = load_data(DATA_PATH)

    # データの前処理
    X, y = preprocess_data(data)

    # データの分割（トレーニングセットとテストセット）
    X_train, X_test, y_train, y_test = split_data(X, y)

    # モデルの学習
    model = train_model(X_train, y_train)

    # 最適なモデルのハイパーパラメータチューニング
    best_model, best_params = hyperparameter_tuning(model, X_train, y_train)
    print("Best Parameters:", best_params)

    # 最適なモデルで評価
    accuracy = evaluate_model(best_model, X_test, y_test)
    print("Best Model Accuracy:", accuracy)

    # モデルの保存
    save_model(best_model)

    # パイプラインの保存
    preprocessing_pipeline = preprocess_data(data)
    save_pipeline(preprocessing_pipeline)

    # モデルとパイプラインの読み込み
    loaded_model = load_saved_model()
    loaded_pipeline = load_saved_pipeline()

    # モデルの再利用
    reuse_model(loaded_model, loaded_pipeline, X_test, y_test)

    # 保存したモデルの評価
    evaluate_saved_model(loaded_model, X_test, y_test)


if __name__ == '__main__':
    main()
