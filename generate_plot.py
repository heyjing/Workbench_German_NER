"""
This python generates plotly figures for csv and txt data.
"""

import pandas as pd
import plotly.express as px


def generate_plotly_figure_from_csv_data(statistic_of_each_tool, options):
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    total_data = []
    tools_name = []
    metrics = []
    for i in statistic_of_each_tool:
        total_data = total_data + list(i['precision']) + list(i['recall']) + list(i['f_value'])
        metrics = metrics + ["precision"] * 4 + ["recall"] * 4 + ["f_value"] * 4
    for j in options:
        tools_name = tools_name + [j] * 12
    df = pd.DataFrame({
        "Entity": ["ORG", "PER", "LOC", "total"] * 3 * len(statistic_of_each_tool),
        "Prozent": total_data,
        "Tools": tools_name,
        "Metrics": metrics
    })
    fig = px.line(df, x="Entity", y="Prozent", color="Tools", line_dash="Metrics")
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return fig


def generate_plotly_figure_from_text_data(statistic_of_each_tool, options):
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    total_data = []
    tools_name = []
    # statistic_of_each_tool is a list that consists of the statistic df of each tool
    for i in statistic_of_each_tool:
        i.drop(i.tail(1).index, inplace=True)
        total_data = total_data + list(i['Number of entities (int)'])
    for j in options:
        tools_name = tools_name + [j] * 3
    df = pd.DataFrame({
        "Entity": ["ORG", "PER", "LOC"] * len(statistic_of_each_tool),
        "Number": total_data,
        "Tools": tools_name
    })
    fig = px.line(df, x="Entity", y="Number", color="Tools")
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig
