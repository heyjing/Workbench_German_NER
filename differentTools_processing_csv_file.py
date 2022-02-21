"""
This python script provides functions to process csv files.

For csv files:

- preprocessing:
    - add_sentence_nr(data)
    - get_sentences(data)

- processing with different models:
    - nltk_analyse_csv_single_sentence(text, entities)
    - spacy_analyse_csv_single_sentence(text, nlp_model, entities)
    - bert_analyse_csv_single_sentence(text, nlp_model, entities)
    - flair_analyse_csv_single_sentence(text, nlp_model, entities)

    - get_result_for_csv_data_type(model_name, sentences, entities, _nlp_model=None) provides a unified interface
      for processing using all of these models

- comparison with ground truth data:
    - get_correct_matches(trained_data, tools_result)
    - statistics(trained_data, tools_result, exec_time)

"""

import streamlit as st
from spacy import displacy
import pandas as pd
import numpy as np
from stqdm import stqdm
import time
import general

@st.experimental_memo(suppress_st_warning=True)
def add_sentence_nr(data):
    """
    This function is for the analysis of csv ground truth data.
    There is only token number in each sentence in the GermanEval 2014 Dataset. This function adds a column named sentence_nr.

    @param data: csv ground truth dataframe

    @return: csv ground truth dataframe with a column named sentence_nr, also called trained_data
    """

    n = 0
    data['sentence_nr'] = None
    # the fourth column is sentence_nr
    for h in stqdm(range(len(data))):
        data.iloc[h, 4] = n
        if h != len(data)-1:
            if data.iloc[h+1, 0] == 1:
                n = n + 1

    return data

@st.experimental_memo(suppress_st_warning=True)
def get_sentences(data):
    """
    This function is for the analysis of csv ground truth data.
    This function extracts the sentences and sentence number from the GermanEval 2014 Dataset and store them in d dataframe.

    @param data: csv ground truth data, the return from add_sentence_nr(data)

    @return: a dataframe that contains each sentence and its sentence number
    """

    df = pd.DataFrame(columns=['sentence_nr', 'sentences'])
    for i in stqdm(range(data['sentence_nr'].max() + 1)):
        sentences = data[data['sentence_nr'] == i]
        sentence = sentences['tokens'].str.cat(sep=' ')
        df = df.append({'sentence_nr': i, 'sentences': sentence}, ignore_index=True)
    df = df.set_index('sentence_nr')
    return df


def spacy_analyse_csv_single_sentence(text, nlp_model, entities):
    """
    This function processes single sentence that is extracted from the csv ground truth data using spacy.
    It returns the found entities of single sentence in a dataframe with two columns 'tokens' and 'label' and
    the html_file for the rendering of this sentence.

    @param text: string
    @param nlp_model: spacy model
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']

    @return: a dataframe contains 'tokens' and 'label' in each sentence and a html string for rendering of each sentence.
    """

    df_single_sentence = pd.DataFrame(columns=['tokens', 'label'])
    ner_results = nlp_model(text)
    html_file = displacy.render(ner_results, style="ent", options=general.choose_entity(entities))
    entity = list(ner_results.ents)
    for i in range(len(entity)):
        df_single_sentence.loc[i] = {'tokens': entity[i].text, 'label': entity[i].label_}
    return df_single_sentence, html_file


@st.experimental_memo(suppress_st_warning=True)
def get_result_for_csv_data_type(model_name, sentences, entities, _nlp_model=None):
    """
    This function provides a unified interface for processing csv ground truth data using different models.

    @param model_name: string
    @param sentences: The dataframe that contains each sentence and its sentence number (return of get_sentences(data))
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']
    @param _nlp_model: optional

    @return: a dataframe contains 'tokens' and 'label' of all sentences and a html string for rendering of all sentences.
    """

    # Because this arg name "nlp_model" is prefixed with "_", it won't be hashed.
    result = pd.DataFrame()
    html_file = []
    start = time.time()
    for i in stqdm(range(len(sentences)), desc="Approximate time needed: "):
        if model_name == "spacy small" or model_name == "spacy middle" or model_name == "spacy large":
            current_sentence_df, single_sentence_html_file = spacy_analyse_csv_single_sentence((str(i) + ". " + sentences.iloc[i]['sentences']), _nlp_model, entities)
        current_sentence_df['sentence_nr'] = i
        result = result.append(current_sentence_df)
        html_file.append(single_sentence_html_file)
    end = time.time()
    exec_time = end - start
    result = (result.set_index(result.columns.drop('tokens', 1).tolist()).tokens.str.split(' ', expand=True).stack().reset_index().rename(
        columns={0: 'tokens'}).loc[:, result.columns])
    return result, exec_time, html_file


def get_correct_matches(trained_data, tools_result):
    """
    This function compares the ground truth data with processing results using different models and returns the correct tokens found by different tools.

    @param trained_data: ground truth dataframe, return of add_sentence_nr(data)
    @param tools_result: The dataframe contains 'tokens' and 'label' of all sentences found by different tools.

    @return: a dataframe contains the correct tokens of their labels of all sentences found by these tools.
    """

    trained_data['tokens_lower'] = trained_data['tokens'].str.lower()
    tools_result['tokens_lower'] = tools_result['tokens'].str.lower()
    merged_df = pd.merge(trained_data, tools_result, on=['sentence_nr', 'tokens_lower'], how='inner')
    conditions = [[x[0] in x[1] for x in zip(merged_df['label'], merged_df['label1'])],
                  [x[0] in x[1] for x in zip(merged_df['label'], merged_df['label2'])]]

    choices = ["LABEL1", "LABEL2"]
    merged_df["Label_Match"] = np.select(conditions, choices, default=np.nan)
    merged_df = merged_df.loc[merged_df['Label_Match'] != "nan"]
    return merged_df


def statistics(trained_data, tools_result, exec_time):
    """
    This function calculates the precision, recall and f_value of different tools and generate a statistic dataframe.

    @param trained_data: ground truth dataframe, return of add_sentence_nr(data)
    @param tools_result: The dataframe contains 'tokens' and 'label' of all sentences found by different tools.
    @param exec_time: processing time by different tools

    @return: a dataframe for statistic
    """

    correct_matches = get_correct_matches(trained_data, tools_result)
    trained_result_with_labels = trained_data.loc[trained_data['label1'] != "O"]

    # Precision is the percentage of named entities found by the learning system that are correct
    precision = len(correct_matches) / len(tools_result)
    # Recall is the percentage of named entities present in the corpus that are found by the system.
    recall = len(correct_matches) / len(trained_result_with_labels)
    # f value:
    beta = 1
    f_value = ((beta * beta + 1) * precision * recall) / (beta * beta * precision + recall)

    # LOC_Statistics
    loc_correct_matches = correct_matches[correct_matches['label'].str.contains('LOC')]
    loc_result = tools_result[tools_result['label'].str.contains('LOC')]
    loc_trained_result = trained_data[trained_data['label1'].str.contains('LOC') | trained_data['label2'].str.contains('LOC')]
    loc_precision = len(loc_correct_matches) / len(loc_result)
    loc_recall = len(loc_correct_matches) / len(loc_trained_result)
    loc_f_value = ((beta * beta + 1) * loc_precision * loc_recall) / (beta * beta * loc_precision + loc_recall)

    # PER_Statistics
    per_correct_matches = correct_matches[correct_matches['label'].str.contains('PER')]
    per_result = tools_result[tools_result['label'].str.contains('PER')]
    per_trained_result = trained_data[
        trained_data['label1'].str.contains('PER') | trained_data['label2'].str.contains('PER')]
    per_precision = len(per_correct_matches) / len(per_result)
    per_recall = len(per_correct_matches) / len(per_trained_result)
    per_f_value = ((beta * beta + 1) * per_precision * per_recall) / (beta * beta * per_precision + per_recall)

    # ORG_Statistics
    org_correct_matches = correct_matches[correct_matches['label'].str.contains('ORG')]
    org_result = tools_result[tools_result['label'].str.contains('ORG')]
    org_trained_result = trained_data[
        trained_data['label1'].str.contains('ORG') | trained_data['label2'].str.contains('ORG')]
    org_precision = len(org_correct_matches) / len(org_result)
    org_recall = len(org_correct_matches) / len(org_trained_result)
    org_f_value = ((beta * beta + 1) * org_precision * org_recall) / (beta * beta * org_precision + org_recall)

    data = {
        'entities': ['ORG', 'PER', 'LOC', 'total'],
        'precision': [round(org_precision, 2), round(per_precision, 2), round(loc_precision, 2), round(precision, 2)],
        'recall': [round(org_recall, 2), round(per_recall, 2), round(loc_recall, 2), round(recall, 2)],
        'f_value': [round(org_f_value, 2), round(per_f_value, 2), round(loc_f_value, 2), round(f_value, 2)],
        'time': [0, 0, 0, round(exec_time, 2)]
    }

    row_labels = [0, 1, 2, 3]
    comparison_result = pd.DataFrame(data=data, index=row_labels)
    return comparison_result
