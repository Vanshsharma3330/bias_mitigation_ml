from sklearn.preprocessing import LabelEncoder

def encode(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df