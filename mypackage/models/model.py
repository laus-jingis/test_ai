import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def save_model(self, directory, filename):
        pass

    @staticmethod
    @abstractmethod
    def load_model(directory, filename):
        pass


class MyModelClass(BaseModel):
    """
    カスタムモデルクラスです。

    Attributes:
        model (keras.models.Sequential): KerasのSequentialモデルオブジェクト

    """

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        """
        モデルの構築を行います。

        Returns:
            keras.models.Sequential: 構築されたSequentialモデルオブジェクト

        """
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=10))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train):
        """
        モデルのトレーニングを行います。

        Args:
            X_train (numpy.ndarray): トレーニングデータの特徴量
            y_train (numpy.ndarray): トレーニングデータのラベル

        Returns:
            None

        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        モデルによる予測を行います。

        Args:
            X_test (numpy.ndarray): テストデータの特徴量

        Returns:
            numpy.ndarray: 予測結果の配列

        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        モデルの評価を行います。

        Args:
            X_test (numpy.ndarray): テストデータの特徴量
            y_test (numpy.ndarray): テストデータのラベル

        Returns:
            list: [損失値, 正確度] の評価結果

        """
        return self.model.evaluate(X_test, y_test)

    def save_model(self, directory, filename):
        """
        モデルを保存します。

        Args:
            directory (str): 保存先ディレクトリのパス
            filename (str): 保存ファイル名

        Returns:
            None

        """
        model_path = os.path.join(directory, filename)
        self.model.save(model_path)

    @staticmethod
    def load_model(directory, filename):
        """
        モデルを読み込みます。

        Args:
            directory (str): モデルが保存されているディレクトリのパス
            filename (str): モデルファイル名

        Returns:
            keras.models.Sequential: 読み込まれたSequentialモデルオブジェクト

        """
        model_path = os.path.join(directory, filename)
        loaded_model = keras.models.load_model(model_path)
        return loaded_model
