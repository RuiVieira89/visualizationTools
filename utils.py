import numpy as np
import pandas as pd


def generate_random_data(lower_lim=1, upper_lim=20, size=(15, 5)):
    data = np.random.normal(size=size)
    columns = np.random.randint(lower_lim, upper_lim, size=size[1])
    df = pd.DataFrame(data, columns=columns)
    return df

def remove_outlier(df, quartile_low=0.2, quartile_high=0.8):

    Q1 = df.quantile(quartile_low)
    Q3 = df.quantile(quartile_high)
    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df
