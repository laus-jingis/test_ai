from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y):
    """
    モデルの予測結果を評価します。

    Args:
        model: 評価対象のモデルオブジェクト。
        X: 評価データの特徴量。
        y: 評価データのラベル。

    Returns:
        float: 正確度（accuracy）スコア。
        float: 適合率（precision）スコア。
        float: 再現率（recall）スコア。
        float: F1スコア。

    """
    # モデルの予測
    y_pred = model.predict(X)

    # 正確度の計算
    accuracy = accuracy_score(y, y_pred)

    # 適合率の計算
    precision = precision_score(y, y_pred)

    # 再現率の計算
    recall = recall_score(y, y_pred)

    # F1スコアの計算
    f1 = f1_score(y, y_pred)

    return accuracy, precision, recall, f1
