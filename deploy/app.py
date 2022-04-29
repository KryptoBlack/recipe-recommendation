import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets, preprocessing

st.title("What's Cooking?")
st.text("Food Recommendation System")
st.image("foood.jpg")

# nav = st.sidebar.radio("Navigation",["Home","IF Necessary 1","If Necessary 2"])

st.subheader("What is your preference?")
vegn = st.radio("Make your selection!", ["veg","non-veg"],index = 1) 

st.subheader("What cuisine do you prefer?")
cuisine = st.selectbox("Choose your favourite!",[ 'Indian', 'French',
       'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai', 'Healthy Food', 'Snack', 'Dessert', 'Japanese'])


st.subheader("Select the desired recipe rating:")  #RATING
val = st.slider("From the least to the best!",0,10)

food = pd.read_csv("../input/food.csv")
ratings = pd.read_csv("../input/ratings.csv")
combined = pd.merge(ratings, food, on='Food_ID')
#ans = food.loc[(food.C_Type == cuisine) & (food.Veg_Non == vegn),['Name','C_Type','Veg_Non']]

ans = combined.loc[(combined["C_Type"] == cuisine) & (combined["Veg_Non"] == vegn) & (combined["Rating"] >= val)]
names = ans['Name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

# print(cuisine, vegn, val)
# print(combined.loc[combined["Veg_Non"] == (vegn)])
finallist = ""
bruh = st.checkbox("Choose your dish:")
if bruh == True:
    finallist = st.selectbox("Select:",ans1)

global dataset
global csr_dataset
##### IMPLEMENTING RECOMMENDER ######
def build_model(params = None):    
    global dataset
    global csr_dataset
    dataset = ans.pivot_table(index='Food_ID',columns='User_ID',values='Rating')
    dataset.fillna(0,inplace=True)
    csr_dataset = csr_matrix(dataset.values)
    dataset.reset_index(inplace=True)
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    return model.fit(csr_dataset)


def food_recommendation(Food_Name):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]
    model = build_model()
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        # try:
        #     Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        # except:
        #     Foodi = 1
        # distances , indices = model.kneighbors(csr_dataset[Foodi], n_neighbors=n+1)    
        # Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        # for val in Food_indices:
        #     Foodi = dataset.iloc[val[0]]['Food_ID']
        #     i = food[food['Food_ID'] == Foodi].index
        #     Recommendations.append({'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})


        Food_indices = ans.drop_duplicates(subset=['Name'], keep='first').sort_values(['Rating'], ascending=False)

        for index, row in Food_indices.iterrows():
            # print(row['Name'])
            # Foodi = row['Food_ID']
            # i = food[food['Food_ID'] == Foodi].index
            if (not row['Name'] == Food_Name):
                Recommendations.append({'Name':row['Name'],'Rating':row['Rating']})
                if Recommendations.__len__() == 10:
                    break
        df = pd.DataFrame(Recommendations)
        return df['Name']
    else:
        return "No Similar Foods."


display = food_recommendation(finallist)

if bruh == True:
    bruh1 = st.checkbox("Our Recommendations: ")
    if bruh1 == True:
        for i in display:
            st.write(i)