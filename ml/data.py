from sklearn.datasets import load_diabetes

def load_dataset(as_frame=True):
    Xy = load_diabetes(as_frame=as_frame)
    frame = Xy.frame
    X = frame.drop(columns=["target"])
    y = frame["target"]  # progression index: higher = worse
    feature_names = list(X.columns)
    return X, y, feature_names
