import numpy as np
import pandas as pd


def calculate_angle(dataframe: pd.DataFrame,
                    target_column: str,
                    diag_column: str,
                    offdiag_column: str,
                    inplace: bool =True):
    '''
    Calculates the angle about the diagonal for a given MEI property
    :param dataframe:
    :param target_column:
    :param diag_column:
    :param offdiag_column:
    :param inplace:
    :return:
    '''
    column = abs(np.arctan(dataframe[offdiag_column] / dataframe[diag_column]))
    if inplace:
        dataframe.insert(dataframe.shape[1], target_column, column)
    else:
        return column
