"""
This python generates user interface and front end using Streamlit components.
"""

import streamlit as st
import pandas as pd
import spacy
import nltk
from transformers import pipeline
from flair.models import SequenceTagger
from stqdm import stqdm

import differentTools_processing_csv_file
import differentTools_processing_txt_file
import general
import generate_plot

st.set_page_config(layout="wide")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stqdm.pandas()

menu = ["CSV data in annotated format", "TXT normal text", "Text input"]
choice = st.sidebar.selectbox("Choose the data format that you want to analyze", menu)


def main():

    if choice == "CSV data in annotated format":
        uploaded_csv_file = st.sidebar.file_uploader(label="Please upload a csv file", type=["csv"])
        st.sidebar.write("This is a sample csv [link](https://drive.google.com/file/d/1juPIuSDPXRZcx7ZX-ZZzmp91g_cuIocA/view?usp=sharing)")

        if uploaded_csv_file is not None:
            st.sidebar.success('Upload success!')
            process_button = st.sidebar.button('Begin Processing')

            if process_button:
                uploaded_text = pd.read_csv(uploaded_csv_file)

                if len(options) > max_model_number:
                    st.warning(
                        'Please choose maximal %s models or change the maximal number of models allowed.' % max_model_number)

                elif len(options) == 0:
                    st.warning('Please choose at least one model.')

                else:
                    with st.spinner('[1/2] Data Preprocessing'):
                        trained_data = differentTools_processing_csv_file.add_sentence_nr(uploaded_text)
                    with st.spinner('[2/2] Data Preprocessing'):
                        sentences = differentTools_processing_csv_file.get_sentences(trained_data)

                    total_statistics, chosen_tools = show_csv_processing(trained_data, sentences)

                    if len(total_statistics) == len(chosen_tools):
                        fig = generate_plot.generate_plotly_figure_from_csv_data(total_statistics, chosen_tools)
                        with st.expander("Data comparison chart"):
                            st.plotly_chart(fig, use_container_width=True)

    if choice == "TXT normal text":
        uploaded_txt_file = st.sidebar.file_uploader(label="Please upload a txt file", type=["txt"])

        if uploaded_txt_file is not None:
            st.sidebar.success('Upload success!')
            process_button = st.sidebar.button('Begin Processing')
            uploaded_text = uploaded_txt_file.read().decode("utf-8")

            if process_button:
                total_statistics, chosen_tools = show_txt_processing(max_model_number, uploaded_text)

                if len(chosen_tools) != 0 and len(total_statistics) == len(chosen_tools):
                    fig = generate_plot.generate_plotly_figure_from_text_data(total_statistics, chosen_tools)
                    with st.expander("Data comparison chart"):
                        st.plotly_chart(fig, use_container_width=True)

    if choice == "Text input":
        texts = st.sidebar.text_area(label="Please enter the texts that you want to analyze:", height=600)
        process_button = st.sidebar.button('Begin Processing')
        if texts:
            if process_button:
                total_statistics, chosen_tools = show_txt_processing(max_model_number, texts)

                if len(chosen_tools) != 0 and len(total_statistics) == len(chosen_tools):
                    fig = generate_plot.generate_plotly_figure_from_text_data(total_statistics, chosen_tools)
                    with st.expander("Data comparison chart"):
                        st.plotly_chart(fig, use_container_width=True)

# APP Layout
st.title("Workbench for German NER Tools")
st.markdown("Please first choose a model or multiple models and then choose your data type on the sidebar.")

options = st.multiselect(
     'Choose a model or multiple models (by default max. 3 models).',
     ['spacy small', 'spacy middle', 'spacy large', 'nltk', 'fhswf/bert_de_ner', 'Davlan/bert-base-multilingual-cased-ner-hrl', 'flair/ner-german', 'flair/ner-german-large', 'flair/ner-multi'])
options1, options2 = st.columns([2, 1])
max_model_number = options1.selectbox(
     'You can change the maximal number of models allowed here',
     (3, 4, 5, 6, 7, 8, 9))

entities_options = options2.multiselect(
     'Choose the entities that you want to visualize',
     ['LOC', 'PER', 'ORG'], default=['LOC', 'PER', 'ORG'])


# sidebar part
st.sidebar.title("File Uploading area")


# @st.experimental_memo
def load_model(model_name):
    if model_name == "spacy small":
        nlp = spacy.load('de_core_news_sm')
    if model_name == "spacy middle":
        nlp = spacy.load('de_core_news_md')
    if model_name == "spacy large":
        nlp = spacy.load('de_core_news_lg')
    if model_name == 'Davlan/bert-base-multilingual-cased-ner-hrl':
        nlp = pipeline(task="ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="average")
    if model_name == 'fhswf/bert_de_ner':
        nlp = pipeline(task="ner", model="fhswf/bert_de_ner", aggregation_strategy="average")
    if model_name == 'flair/ner-german':
        nlp = SequenceTagger.load("flair/ner-german")
    if model_name == 'flair/ner-german-large':
        nlp = SequenceTagger.load("flair/ner-german-large")
    if model_name == 'flair/ner-multi':
        nlp = SequenceTagger.load("flair/ner-multi")
    return nlp


def csv_ground_truth_data_processing(model_name, trained_data, sentences, col_statistik, col_annotation):
    """
    1. This function receives the user input (model_name, entities_options) and passes parameters the to back end.
    2. After processing using different models, it compares the result with the ground truth data.
    3. At last, it shows statistic and renders html strings in the corresponding fields.

    @param model_name: string
    @param trained_data: ground truth dataframe, return of add_sentence_nr(data)
    @param sentences: The dataframe that contains each sentence and its sentence number (return of get_sentences(data))
    @param col_statistik: Streamlit containers
    @param col_annotation: Streamlit containers

    @return: a dataframe for statistic of one model
    """

    if model_name != "nltk":
        with st.spinner('[1/2] Loading ' + model_name + ' model...'):
            nlp_model = load_model(model_name)

        with st.spinner('[2/2] Wait for ' + model_name + ' model identifying named entities'):
            result, time, html_file_stored_in_list = differentTools_processing_csv_file.get_result_for_csv_data_type(model_name, sentences, entities_options, nlp_model)

    if model_name == "nltk":
        result, time, html_file_stored_in_list = differentTools_processing_csv_file.get_result_for_csv_data_type(model_name, sentences,
                                                                                                                 entities_options)
    statistics = differentTools_processing_csv_file.statistics(trained_data, result, time)
    statistics_with_color = statistics.style.apply(general.statistic_df_row_style, axis=1).set_precision(
        2)
    col_statistik.write(statistics_with_color)

    with col_annotation.expander(model_name + " Visualisation"):
        for i in html_file_stored_in_list:
            st.markdown(i, unsafe_allow_html=True)

    return statistics


# Fill normal text statistic and visualization data
def normal_text_processing(model_name, text, col_statistik, col_annotation):
    """
    1. This function receives the user input (model_name, entities_options) and passes parameters the to back end.
    2. After processing using different models, it generates statistics and html strings for the rendering.
    3. At last, it shows statistic and renders html strings in the corresponding fields.

    @param model_name: string
    @param text: string
    @param col_statistik: Streamlit containers
    @param col_annotation: Streamlit containers

    @return: a dataframe for statistic of one model
    """

    if model_name != "nltk":
        with st.spinner('[1/2] Loading ' + model_name + ' model...'):
            nlp_model = load_model(model_name)

        with st.spinner('[2/2] Wait for ' + model_name + ' model identifying named entities'):
            df_of_each_tool, html_file = differentTools_processing_txt_file.analyse_normal_text(model_name, text,
                                                                                                entities_options, nlp_model)
    if model_name == "nltk":
        df_of_each_tool, html_file = differentTools_processing_txt_file.analyse_normal_text(model_name, text,
                                                                                            entities_options)

    df_with_color = df_of_each_tool.style.apply(general.statistic_df_row_style, axis=1).set_precision(2)
    col_statistik.write(df_with_color)

    with col_annotation.expander(model_name + " Visualisation"):
        for i in html_file:
            st.markdown(i, unsafe_allow_html=True)

    return df_of_each_tool


def show_csv_processing(trained_data, sentences):
    """
    Assign containers to results of different models based on the maximum number of models allowed and
    the number of models actually selected.

    @param trained_data: ground truth dataframe, return of add_sentence_nr(data)
    @param sentences: The dataframe that contains each sentence and its sentence number (return of get_sentences(data))

    @return: a single big dataframe of statistic of all the selected models
    """

    total_statistics = list()

    if len(options) == 1:
        result_tool0 = csv_ground_truth_data_processing(options[0], trained_data, sentences, st, st)
        total_statistics.append(result_tool0)
    if len(options) >= 2:
        layout_when_more_than_two_models_selected("csv", len(options), total_statistics, trained_data=trained_data, sentences=sentences)

    return total_statistics, options


def show_txt_processing(max_model_num, text):
    """
    Assign containers to results of different models based on the maximum number of models allowed and
    the number of models actually selected.

    @param max_model_num: string
    @param text: string

    @return: a single big dataframe of statistic of all the selected models
    """

    total_statistics = list()
    if len(options) > max_model_num:
        st.warning('Please choose maximal %s models or change the maximal number of models allowed.' % max_model_num)
    elif len(options) == 0:
        st.warning('Please choose at least one model.')
    elif len(options) == 1:
        result_tool0 = normal_text_processing(options[0], text, st, st)
        total_statistics.append(result_tool0)
    elif len(options) >= 2:
        layout_when_more_than_two_models_selected("txt", len(options), total_statistics, text=text)

    return total_statistics, options


def layout_when_more_than_two_models_selected(data_type, model_number, total_statistics, trained_data=None, sentences=None, text=None):
    """
    Assign containers to results of different models when more than two models selected.

    @param data_type: "csv" or "txt"
    @param model_number: integer
    @param total_statistics: an empty list
    @param trained_data: optional, ground truth dataframe, return of add_sentence_nr(data)
    @param sentences: optional, the dataframe that contains each sentence and its sentence number (return of get_sentences(data))
    @param text: optional, string
    """

    column_title = st.columns(model_number)
    column_progress_bar = st.columns(model_number)
    column_statistik = st.columns(model_number)
    column_annotation = st.columns(model_number)
    for i in range(model_number):
        column_title[i].markdown("### " + options[i])
        with column_progress_bar[i]:
            if data_type == "csv":
                result_tool = csv_ground_truth_data_processing(options[i], trained_data, sentences, column_statistik[i], column_annotation[i])
            if data_type == "txt":
                result_tool = normal_text_processing(options[i], text, column_statistik[i], column_annotation[i])
        total_statistics.append(result_tool)

if __name__ == '__main__':
    main()
