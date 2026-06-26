import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):

    df.drop("id", axis=1, inplace=True)

    df["bmi"] = df["bmi"].fillna(
    df["bmi"].median()
    )

    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df