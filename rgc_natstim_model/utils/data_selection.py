import pandas as pd
from copy import deepcopy

def filter_df(df, keywords, criteria, relations, axis,
              restrict):
    """
    Takes a dataframe as input, together with a list of keywords and
    corresponding
    criteria to apply to the data in the columns indicated by the keywords
    :param df:
    :param keywords:
    :param criteria:
    :return:
    """
    conditions = []
    for kw, crit, rel in zip(keywords, criteria, relations):
        if rel == "<":
            condition = df[kw] < crit
        elif rel == "<=":
            condition = df[kw] <= crit
        elif rel == ">":
            condition = df[kw] > crit
        elif rel == ">=":
            condition = df[kw] >= crit
        elif rel == "==":
            condition = df[kw] == crit
        elif rel == "!=":
            condition = df[kw] != crit
        else:
            raise UserWarning('{} is not a valid relation'.format(rel))
        conditions.append(condition)
    if not restrict:  # if not restrict, then concat
        out_df = pd.concat([df[cond] for cond in conditions],
                     join="inner", axis=axis)
    elif restrict:
        out_df = deepcopy(df)
        for cond in conditions:
            out_df = out_df.loc[cond]
    return out_df