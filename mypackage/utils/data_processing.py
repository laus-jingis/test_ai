from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    """
    データの前処理を行います。

    Args:
        data (pandas.DataFrame): 前処理するデータフレーム。

    Returns:
        tuple: 前処理された特徴量とラベルのタプル (X_scaled, y)。

    """
    # 特徴量とラベルの分割
    X = data.drop('label', axis=1)
    y = data['label']

    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
