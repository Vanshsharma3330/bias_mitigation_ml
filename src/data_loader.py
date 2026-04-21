import pandas as pd

def load_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    df = df.dropna()

    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
    df["sex"] = df["sex"].apply(lambda x: 1 if x == "Male" else 0)

    return df


def load_compas():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    
    df = pd.read_csv(url)

    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ]

    df["race"] = df["race"].apply(lambda x: 1 if x == "African-American" else 0)
    df["sex"] = df["sex"].apply(lambda x: 1 if x == "Male" else 0)

    df = df[[
        "age", "priors_count", "race", "sex",
        "two_year_recid"
    ]]

    return df