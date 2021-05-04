from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def run_knn(X_train, y_train, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    return k

def pixels_conversion(filename, nm=False):
    PIXEL_TO_MICRON_SCALAR = (1 / 1789.58611481976)
    PIXEL_TO_NANOMETER_SCALAR = 1

    try:
        df = pd.read_csv(filename, delimiter=",")
        df.drop(df.columns[[0, 2]], axis=1, inplace=True)
        if nm:
            df['X'] = df['X'] * PIXEL_TO_NANOMETER_SCALAR
            df['Y'] = df['Y'] * PIXEL_TO_NANOMETER_SCALAR
        else:
            df['X'] = df['X'] * PIXEL_TO_MICRON_SCALAR
            df['Y'] = df['Y'] * PIXEL_TO_MICRON_SCALAR
        df['X'] = df['X'].round(3)
        df['Y'] = df['Y'].round(3)
        print(df.head())
        return df
    except Exception as e:
        print(e)

