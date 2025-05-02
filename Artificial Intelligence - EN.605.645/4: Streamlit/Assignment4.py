#import necessary libraries
import streamlit as st
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
from operator import itemgetter

#parse data file and extract data
def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


#calculate the euclidean distance between two points
def euclidean_distance(datapoint: list[float], query: list[float]) -> float:
    squared_distance = 0
    for i in range(len(datapoint) - 1):  # assuming last column is the target variable
        difference = (datapoint[i] - query[i]) ** 2
        squared_distance = squared_distance + difference
    return math.sqrt(squared_distance)


#find the k-nearest neighbors of a point and calculate the prediction
def knn_predict(train: list[list[float]], test: list[float], k: int) -> float:
    distances = {}
    for train_row in train:
        distance = euclidean_distance(train_row, test)
        distances[tuple(train_row)] = distance

    top_k = dict(sorted(distances.items(),
                 key=itemgetter(1))[:k])

    prediction = 0
    for key in top_k:
        prediction = prediction + key[-1]
    prediction = prediction / k

    return top_k, prediction


#find the minimum and maximum of possible values that can be inputted by user
#double the range by subtracting 50% of the range from min, and adding 50% of the range to max
def slider_min_max(data, header):
    for i, item in enumerate(header):
        min = np.min(data[:, i], axis=0)
        max = np.max(data[:, i], axis=0)
        difference = max - min
        min = min - (0.5 * difference)
        if min < 0:
            min = 0
        max = max + (0.5 * difference)
        mid = int(min + (0.5*(max - min)))
        header[i] = [item, float(min), float(max), float(mid)]
        print(header)
    return header


data = parse_data("concrete_compressive_strength.csv")

data = np.array(data)

column_names = ['Cement', 'Slag', 'Ash', 'Water', 'Superplasticizer',
                'Coarse Aggregate', 'Fine Aggregate', 'Age']

min_max = slider_min_max(data, column_names)

#text on the main page
st.title("Compressive Strength of Concrete")
st.header(
    "Use k-nearest neighbors to predict the compressive strength (MPa) of your concrete")

st.divider()

k_value = st.slider("Select a k value:", 1, 21, 9)

st.divider()

st.sidebar.header("Properties of your concrete:")

#input sliders to adjust the input parameters
cement = st.sidebar.slider(
    "Cement: kg in a cubic meter mixture", min_max[0][1], min_max[0][2], value=min_max[0][3])

st.sidebar.divider()

slag = st.sidebar.slider(
    "Slag: kg in a cubic meter mixture", min_max[1][1], min_max[1][2], value=min_max[1][3])

st.sidebar.divider()

ash = st.sidebar.slider(
    "Ash: kg in a cubic meter mixture", min_max[2][1], min_max[2][2], value=min_max[2][3])

st.sidebar.divider()

water = st.sidebar.slider(
    "Water: kg in a cubic meter mixture", min_max[3][1], min_max[3][2], value=min_max[3][3])

st.sidebar.divider()

superplasticizer = st.sidebar.slider(
    "Superplasticizer: kg in a cubic meter mixture", min_max[4][1], min_max[4][2], value=min_max[4][3])

st.sidebar.divider()

coarse_aggregate = st.sidebar.slider(
    "Coarse Aggregate: kg in a cubic meter mixture", min_max[5][1], min_max[5][2], value=min_max[5][3])

st.sidebar.divider()

fine_aggregate = st.sidebar.slider(
    "Fine Aggregate: kg in a cubic meter mixture", min_max[6][1], min_max[6][2], value=min_max[6][3])

st.sidebar.divider()

age = st.sidebar.slider(
    "Age: kg in a cubic meter mixture", min_max[7][1], min_max[7][2], value=min_max[7][3])

#aggregate slider input as a test point
test_point = [cement, slag, ash, water, superplasticizer,
              coarse_aggregate, fine_aggregate, age]


#print input data as a table
st.write("Your Concrete Composition:")

st.dataframe(
    np.array([test_point]),
    column_config={
        "0": "Cement",
        "1": "Slag",
        "2": "Ash",
        "3": "Water",
        "4": "Superplasticizer",
        "5": "Coarse Aggregate",
        "6": "Fine Aggregate",
        "7": "Age",
    },
    hide_index=True,
)

st.divider()

#make a prediction
top_k, prediction = knn_predict(data, test_point, k_value)

st.subheader("Predicted Compression Strength: " +
             str(round(prediction, 5)) + " MPa")

st.divider()

# labels to use for graphing the k-nearest neighbors
neighbor_labels = []
for i in range(1, k_value + 1):
    string = "Neighbor " + str(i)
    neighbor_labels.append(string)

top_k_list = []
i = 0
for key in top_k:
    key = list(key)
    key.append(top_k[tuple(key)])
    key.append(neighbor_labels[i])
    top_k_list.append(key)
    i = i + 1

top_k_array = np.array(top_k_list)

#create a table of the k-nearest neighbors
with st.expander("K-Nearest Neighbors"):
    st.write("K-Nearest Neighbors: from closest to furthest")
    # output nearest neighbors table
    st.dataframe(
        top_k_array,
        column_config={
            "0": "Cement",
            "1": "Slag",
            "2": "Ash",
            "3": "Water",
            "4": "Superplasticizer",
            "5": "Coarse Aggregate",
            "6": "Fine Aggregate",
            "7": "Age",
            "8": "Compression Strength",
            "9": "Euclidean Distance",
            "10": "Neighbor"
        },
        column_order=("10", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
        hide_index=True,
    )

test_point.append(prediction)
test_point.append(0)
test_point.append("Your Concrete")
top_k_array = np.vstack([top_k_array, test_point])

plot_data = {top_k_array[i, 10]: float(top_k_array[i, 8])
             for i in range(len(top_k_array[:, 10]))}

plot_data = dict(sorted(plot_data.items(), key=lambda x: x[1]))

#plot bar charts to show how the test point compares to the nearest neighbors
with st.expander("Comparing Your Neighbors"):

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["Compressive Strength", "Cement", "Slag", "Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"])

    with tab1:
        tab1_data = {top_k_array[i, 10]: float(top_k_array[i, 8])
                     for i in range(len(top_k_array[:, 10]))}

        tab1_data = dict(sorted(tab1_data.items(), key=lambda x: x[1]))
        st.subheader("Comparing Compression Strength")
        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab1_data.keys()]
        ax.barh(list(tab1_data.keys()), list(
            tab1_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Compression Strength (MPa)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Compression Strength of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab2:
        st.subheader("Comparing Cement Composition")

        tab2_data = {top_k_array[i, 10]: float(top_k_array[i, 0])
                     for i in range(len(top_k_array[:, 10]))}
        tab2_data = dict(sorted(tab2_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab2_data.keys()]
        ax.barh(list(tab2_data.keys()), list(
            tab2_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Cement (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Cement Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab3:
        st.subheader("Comparing Slag Composition")

        tab3_data = {top_k_array[i, 10]: float(top_k_array[i, 1])
                     for i in range(len(top_k_array[:, 10]))}
        tab3_data = dict(sorted(tab3_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab3_data.keys()]
        ax.barh(list(tab3_data.keys()), list(
            tab3_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Slag (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Slag Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab4:
        st.subheader("Comparing Ash Composition")

        tab4_data = {top_k_array[i, 10]: float(top_k_array[i, 2])
                     for i in range(len(top_k_array[:, 10]))}
        tab4_data = dict(
            sorted(tab4_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab4_data.keys()]
        ax.barh(list(tab4_data.keys()), list(
            tab4_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Ash (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Ash Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab5:
        st.subheader("Comparing Water Composition")

        tab5_data = {top_k_array[i, 10]: float(top_k_array[i, 3])
                     for i in range(len(top_k_array[:, 10]))}
        tab5_data = dict(
            sorted(tab5_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab5_data.keys()]
        ax.barh(list(tab5_data.keys()), list(
            tab5_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Water (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Water Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab6:
        st.subheader("Comparing Superplasticizer Composition")

        tab6_data = {top_k_array[i, 10]: float(top_k_array[i, 4])
                     for i in range(len(top_k_array[:, 10]))}
        tab6_data = dict(
            sorted(tab6_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab6_data.keys()]
        ax.barh(list(tab6_data.keys()), list(
            tab6_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Superplasticizer (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Superplasticizer Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab7:
        st.subheader("Comparing Coarse Aggregate Composition")

        tab7_data = {top_k_array[i, 10]: float(top_k_array[i, 5])
                     for i in range(len(top_k_array[:, 10]))}
        tab7_data = dict(
            sorted(tab7_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab7_data.keys()]
        ax.barh(list(tab7_data.keys()), list(
            tab7_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Coarse Aggregate (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Coarse Aggregate Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab8:
        st.subheader("Comparing Fine Aggregate Composition")

        tab8_data = {top_k_array[i, 10]: float(top_k_array[i, 6])
                     for i in range(len(top_k_array[:, 10]))}
        tab8_data = dict(
            sorted(tab8_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab8_data.keys()]
        ax.barh(list(tab8_data.keys()), list(
            tab8_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Fine Aggregate (kg in cubic meter)')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Fine Aggregate Composition of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)

    with tab9:
        st.subheader("Comparing Age Composition")

        tab9_data = {top_k_array[i, 10]: float(top_k_array[i, 7])
                     for i in range(len(top_k_array[:, 10]))}
        tab9_data = dict(
            sorted(tab9_data.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots()
        colors = ["red" if i ==
                  'Your Concrete' else "black" for i in tab9_data.keys()]
        ax.barh(list(tab9_data.keys()), list(
            tab9_data.values()), align='center', color=colors)
        ax.set_xlim(xmin=0)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Age')
        ax.set_ylabel('k-Nearest Neighbors')
        ax.set_title(
            'Age of Your Concrete and its k-Nearest Neighbors')

        st.pyplot(fig)
