"""
This python script provides functions to process txt files.

For txt files:

- processing with different models:
    - nltk_analyse_normal_text(text, entities)
    - spacy_analyse_normal_text(text, nlp_model, entities)
    - bert_analyse_normal_text(text, nlp_model, entities)
    - flair_analyse_normal_text(text, nlp_model, entities)

    - analyse_normal_text(model_name, text, entities, _nlp_model=None) provides a unified interface
      for processing txt data using these models

- Other supporting function:
    - show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time)

"""

from spacy import displacy
import pandas as pd
import nltk
from flair.data import Sentence
from stqdm import stqdm
from nltk import Tree
import time
import re
import general


def show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time):
    """
    This function generates a statistic dataframe for txt data types.

    @param org_number: integer
    @param per_number: integer
    @param loc_number: integer
    @param exec_time: the number of seconds passed

    @return: a dataframe.
    """

    data = {
        'entities': ['ORG', 'PER', 'LOC', 'time'],
        'Number of entities (int)': [org_number, per_number, loc_number, round(exec_time, 2)]
    }
    row_labels = [0, 1, 2, 3]
    result = pd.DataFrame(data=data, index=row_labels)
    return result


def nltk_analyse_normal_text(text, entities):
    """
    This function process text type data using nltk.

    @param text: string
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']

    @return: a dataframe for statistik and a html string for rendering.
    """

    html_file = []
    text = re.sub(' +', ' ', text)
    text_split_stored_in_list = text.strip().split('\r\n')
    # remove empty strings from list
    text_split_stored_in_list = list(filter(None, text_split_stored_in_list))

    start = time.time()
    org_number = 0
    per_number = 0
    loc_number = 0

    for i in stqdm(range(len(text_split_stored_in_list))):
        ner_results = []
        for sent in nltk.sent_tokenize(text_split_stored_in_list[i], language='german'):
            for chunk in nltk.chunk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if isinstance(chunk, Tree):
                    if hasattr(chunk, 'label'):
                        current_token = ' '.join([w for w, _ in chunk.leaves()])

                        if chunk.label() == "GPE":
                            ner_results.append(
                                {"start": text_split_stored_in_list[i].find(current_token),
                                 "end": text_split_stored_in_list[i].find(current_token) + len(current_token),
                                 "label": "LOC"}
                            )
                            loc_number += 1
                        if chunk.label() != "GPE":
                            ner_results.append(
                                {"start": text_split_stored_in_list[i].find(current_token),
                                 "end": text_split_stored_in_list[i].find(current_token) + len(current_token),
                                 "label": chunk.label()[0:3]}
                            )
                            if chunk.label() == "PERSON":
                                per_number += 1
                            if chunk.label() == "ORGANIZATION":
                                org_number += 1
                            if chunk.label() == "LOCATION":
                                loc_number += 1

        rendering_format = [{"text": text_split_stored_in_list[i], "ents": ner_results, "title": None}]

        html_file.append(
            displacy.render(rendering_format, style="ent", options=general.choose_entity(entities), manual=True))

    end = time.time()
    exec_time = end - start
    df_statistic = show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time)

    return df_statistic, html_file


def spacy_analyse_normal_text(text, nlp_model, entities):
    """
    This function process text type data using spacy.

    @param text: string
    @param nlp_model: spacy model
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']

    @return: a dataframe for statistik and a html string for rendering.
    """

    html_file = []
    org_number: int = 0
    per_number = 0
    loc_number = 0
    text = re.sub(' +', ' ', text)
    text_split_stored_in_list = text.strip().split('\r\n')
    # remove empty strings from list
    text_split_stored_in_list = list(filter(None, text_split_stored_in_list))

    start = time.time()

    docs = nlp_model.pipe(text_split_stored_in_list, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    for doc in stqdm(docs, total=len(text_split_stored_in_list)):
        html_file.append(displacy.render(doc, style="ent", options=general.choose_entity(entities)))
        entity = list(doc.ents)
        for i in range(len(entity)):
            if entity[i].label_ == "ORG":
                org_number += 1
            if entity[i].label_ == "PER":
                per_number += 1
            if entity[i].label_ == "LOC":
                loc_number += 1

    end = time.time()
    exec_time = end - start

    df_statistic = show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time)

    return df_statistic, html_file


def bert_analyse_normal_text(text, nlp_model, entities):
    """
    This function process text type data using bert.

    @param text: string
    @param nlp_model: bert model
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']

    @return: a dataframe for statistik and a html string for rendering.
    """

    html_file = []
    # remove whitespaces in the beginning and in the end of a string
    text = re.sub(' +', ' ', text)
    text_split_stored_in_list = text.strip().split('\r\n')
    # remove empty strings from list
    text_split_stored_in_list = list(filter(None, text_split_stored_in_list))

    org_number: int = 0
    per_number = 0
    loc_number = 0

    start = time.time()

    for i in stqdm(range(len(text_split_stored_in_list))):

        ner_results = nlp_model(text_split_stored_in_list[i])

        for j in ner_results:
            j["label"] = j.pop("entity_group")
            if j["label"] == "LOC":
                loc_number += 1
            if j["label"] == "ORG":
                org_number += 1
            if j["label"] == "PER":
                per_number += 1
        rendering_format = [{"text": text_split_stored_in_list[i], "ents": ner_results, "title": None}]

        html_file.append(
            displacy.render(rendering_format, style="ent", options=general.choose_entity(entities), manual=True))

    end = time.time()
    exec_time = end - start

    df_statistic = show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time)

    return df_statistic, html_file


def flair_analyse_normal_text(text, nlp_model, entities):
    """
    This function process text type data using flair.

    @param text: string
    @param nlp_model: flair model
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']

    @return: a dataframe for statistik and a html string for rendering.
    """

    html_file = []
    text = re.sub(' +', ' ', text)
    text_split_stored_in_list = text.strip().split('\r\n')
    # remove empty strings from list
    text_split_stored_in_list = list(filter(None, text_split_stored_in_list))

    org_number: int = 0
    per_number = 0
    loc_number = 0

    start = time.time()

    for i in stqdm(range(len(text_split_stored_in_list))):
        sentence = Sentence(text_split_stored_in_list[i])
        nlp_model.predict(sentence)
        ner_results = []

        for entity in sentence.get_spans('ner'):
            ner_results.append({"start": entity.start_pos, "end": entity.end_pos, "label": str(entity.labels)[1:4]})

            if str(entity.labels)[1:4] == "PER":
                per_number += 1
            if str(entity.labels)[1:4] == "LOC":
                loc_number += 1
            if str(entity.labels)[1:4] == "ORG":
                org_number += 1
        rendering_format = [{"text": text_split_stored_in_list[i], "ents": ner_results, "title": None}]
        html_file.append(displacy.render(rendering_format, style="ent", options=general.choose_entity(entities), manual=True))

    end = time.time()
    exec_time = end - start

    df_statistic = show_number_of_entities_in_df(org_number, per_number, loc_number, exec_time)

    return df_statistic, html_file


@st.experimental_memo(suppress_st_warning=True)
def analyse_normal_text(model_name, text, entities, _nlp_model=None):
    """
    This function provides a unified interface for processing text type data using different models.

    @param model_name: string
    @param text: string
    @param entities: list contains of different entity type, such as ['LOC', 'PER', 'ORG']
    @param _nlp_model: optional

    @return: a dataframe for statistik and a html string for rendering.
    """

    if model_name == "spacy small" or model_name == "spacy middle" or model_name == "spacy large":
        df_statistic, html_file = spacy_analyse_normal_text(text, _nlp_model, entities)
    if model_name == "Davlan/bert-base-multilingual-cased-ner-hrl" or model_name == "fhswf/bert_de_ner":
        df_statistic, html_file = bert_analyse_normal_text(text, _nlp_model, entities)
    if model_name == "flair/ner-german" or model_name == "flair/ner-german-large" or model_name == "flair/ner-multi":
        df_statistic, html_file = flair_analyse_normal_text(text, _nlp_model, entities)
    if model_name == "nltk":
        df_statistic, html_file = nltk_analyse_normal_text(text, entities)
    return df_statistic, html_file

