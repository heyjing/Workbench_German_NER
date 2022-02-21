"""
This python script provides supporting functions to process csv or txt files.
General:
    - choose_entity(ents_list)
    - statistic_df_row_style(row)
"""

import pandas as pd


def choose_entity(ents_list):
    """
    This function is for visualisation. The argument ents_list controls which entity type will be shown in the visualisation.
    This function defines the colors for 'LOC', 'PER', 'ORG' entity type. The return will serve as options by the visualisation.

    @param ents_list: an entity list, for example ['LOC', 'PER', 'ORG']

    @return: a dictionary.
    """

    entity_color_options = {"ents": ents_list,
                            "colors": {"ORG": "#fea", "PER": "#8ef", "LOC": "#faa"}}
    return entity_color_options


def statistic_df_row_style(row):
    """
    This function defines the color of the statistic dataframes for csv type and txt type data.
    """

    if row['entities'] == 'ORG':
        return pd.Series('background-color: #fea', row.index)
    if row['entities'] == 'PER':
        return pd.Series('background-color: #8ef', row.index)
    if row['entities'] == 'LOC':
        return pd.Series('background-color: #faa', row.index)
