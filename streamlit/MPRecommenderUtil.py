import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from surprise import SVD, KNNBasic,KNNWithMeans, SVDpp, NMF, BaselineOnly
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split


def get_predictions(user_id):
    preds = []
    for route in routes['route_id'].unique():
        preds.append((route,svd.predict(user_id, route).est))
    return  zip(*sorted(preds,key = lambda x: x[1],reverse = True))


def get_top_n(user_id, n= 10, area_ids = [], climb_types = [], min_grade = 0, max_grade = 71, pitches = 1, ignore_completed = True):
    preds = pd.DataFrame(list(zip(*get_predictions(user_id))), columns = ['route_id', 'prediction'])

    
    user_preds = routes.merge(preds, on = 'route_id')
    
    if ignore_completed:
        already_rated = ratings[ratings['user_id'] == user_id]['route_id'].values

        user_preds = user_preds[~user_preds['route_id'].isin(already_rated)]

    if climb_types == []:
        climb_types = ['Boulder', 'Sport', 'Trad']

    if area_ids == []:
        return user_preds[(user_preds['type'].isin(climb_types)) & (user_preds['grade_numeric'].isin(range(min_grade, max_grade+1))) & (user_preds['pitches'].isin(list(range(pitches, 100 if pitches != 1 else 2))))].sort_values('prediction', ascending = False).head(n)
    else:
        subareas = functools.reduce(operator.iconcat, [area_tree.get_children(area_id) for area_id in area_ids], [])
        return user_preds[(user_preds['type'].isin(climb_types)) & (user_preds['area_id'].isin(subareas)) & (user_preds['pitches'].isin(list(range(pitches,100 if pitches != 1 else 2)))) & (user_preds['grade_numeric'].isin(list(range(min_grade, max_grade+1))))].sort_values('prediction', ascending = False).head(n)

     #   result = pd.DataFrame()
      #  i = 0
      #  count = 0
      #  rewts = []
      #  for i, route in enumerate(top_routes):           
       #     if routes[routes['route_id'] == route]['type'].iloc[0] in climb_types:
       #         rewts.append(route)
       #     if len(rewts) >= n:
       #         break
       # return routes[routes['route_id'].isin(rewts)]