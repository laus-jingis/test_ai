import os
import pandas as pd


def load_data(data_path):
    """
    指定されたデータパスからデータを読み込みます。

    Args:
        data_path (str): データファイルのパス。

    Returns:
        pandas.DataFrame: 読み込まれたデータを格納したデータフレーム。

    """
    # データの読み込み
    data = pd.read_csv(data_path)
    return data
