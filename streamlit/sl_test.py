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


import pickle

import functools
import operator 
#from MPRecommenderUtil import get_top_n

st.set_page_config(layout="wide")
mp_area_url = 'https://www.mountainproject.com/area/'
mp_route_url = 'https://www.mountainproject.com/route/'
class MPAreaTree:
    def __init__(self, areas = None):
        if(areas is not None):
            self.build(areas)            
        else:
            self.area_dict = {}
            self.areas = areas.copy()
            
    def build(self, areas):
        self.area_dict = {}
        self.areas = areas.copy()
        for i,row in areas.iterrows():
            self.area_dict[row['area_id']] = {'area_name' : row.area_name, 'parent' : row.parent_id, 'children' : []}
        for i,row in areas.iterrows():
            if row["parent_id"] != 0:
                self.area_dict[row["parent_id"]]['children'].append(row["area_id"])
                
    def get_name(self, area_id):
        return self.areas[self.areas['area_id'] == area_id]['area_name'].unique()[0]

    def get_parent_chain(self, area_id):
        chain = []
        current_id = area_id
        while current_id != 0:
            chain = [(self.get_name(current_id), current_id)] + chain
            current_id = self.area_dict[current_id]['parent']
        return chain
    def get_children(self, area_id):
        
        return [area_id] + functools.reduce(operator.iconcat, [self.get_children(child) for child in self.area_dict[area_id]['children']], [])
    def get_parent_chain_names(self, area_id):
        chain = get_parent_chain(area_id)
        return [self.get_name(x) for x in chain] 
    
    def get_height(self,area_id):
        if len(self.area_dict[area_id]['children']) == 0:
            return 0
        else:
            return 1 + max([self.get_height(child) for child in self.area_dict[area_id]['children']]) 

    def get_formatted_name(self, area_id):
        if self.area_dict[area_id]['parent'] == 0:
            return self.area_dict[area_id]['area_name']
        return self.get_formatted_name(self.area_dict[area_id]['parent']) + ' > ' + self.area_dict[area_id]['area_name']
    
    
    def get_depth(self,area_id):
        return len(self.get_parent_chain(area_id))-1
    
    def get_link_chain(self, area_id):
        if self.area_dict[area_id]['parent'] == 0:
            return f'<a target="_blank" href="{mp_area_url}{area_id}">{self.area_dict[area_id]["area_name"]}</a>'
        
        return self.get_link_chain(self.area_dict[area_id]["parent"]) + ' > ' + f'<a target="_blank" href="{mp_area_url}{area_id}">{self.area_dict[area_id]["area_name"]}</a>'
    def get_area_name(self, area_id):
    	return self.area_dict[area_id]['area_name']

def get_predictions(user_id):
    preds = []
    for route in routes['route_id'].unique():
        preds.append((route,svd.predict(user_id, route).est))
    return  zip(*sorted(preds,key = lambda x: x[1],reverse = True))


def get_top_n(user_id, n= 10, area_ids = [], climb_types = [], min_grade = 0, max_grade = 71, pitch_range = range(1,100), ignore_completed = True):
    preds = pd.DataFrame(list(zip(*get_predictions(user_id))), columns = ['route_id', 'prediction'])

    
    user_preds = routes.merge(preds, on = 'route_id')
    
    if ignore_completed:
        already_rated = ratings[ratings['user_id'] == user_id]['route_id'].values
        user_preds = user_preds[~user_preds['route_id'].isin(already_rated)]

    if climb_types == []:
        climb_types = ['Boulder', 'Sport', 'Trad']

    if area_ids == []:
        return user_preds[(user_preds['type'].isin(climb_types)) & (user_preds['grade_numeric'].isin(range(min_grade, max_grade+1))) & (user_preds['pitches'].isin(list(range(pitches, 100 if pitches != 1 else 2))))].sort_values('prediction', ascending = False).head(n).drop(columns = ['description', 'star_ratings'])
    else:
        subareas = functools.reduce(operator.iconcat, [area_tree.get_children(area_id) for area_id in area_ids], [])
        return user_preds[(user_preds['type'].isin(climb_types)) & (user_preds['area_id'].isin(subareas)) & (user_preds['pitches'].isin(list(range(pitches,100 if pitches != 1 else 2)))) & (user_preds['grade_numeric'].isin(list(range(min_grade, max_grade+1))))].sort_values('prediction', ascending = False).head(n).drop(columns = ['description', 'star_ratings'])

def get_predictions(user_id):
    preds = []
    for route in routes['route_id'].unique():
        preds.append((route,svd.predict(user_id, route).est))
    return  zip(*sorted(preds,key = lambda x: x[1],reverse = True))


def get_top_n(user_id, n= 10, area_ids = [], climb_types = [], min_climb_grade = 0, max_climb_grade = 71, min_boulder_grade = 0, max_boulder_grade = 71, pitch_range = range(1,100), ignore_completed = True):
    preds = pd.DataFrame(list(zip(*get_predictions(user_id))), columns = ['route_id', 'prediction'])

    
    user_preds = routes.merge(preds, on = 'route_id')
    
    if ignore_completed:
        already_rated = ratings[ratings['user_id'] == user_id]['route_id'].values
        user_preds = user_preds[~user_preds['route_id'].isin(already_rated)]

    if climb_types == []:
        climb_types = ['Boulder', 'Sport', 'Trad']

    boulder_mask = (user_preds['type'] == 'Boulder') & (user_preds['grade_numeric'].isin(range(min_boulder_grade, max_boulder_grade+1))) if 'Boulder' in climb_types else False
    sport_mask = (user_preds['type'] == 'Sport') & (user_preds['grade_numeric'].isin(range(min_climb_grade, max_climb_grade+1))) if 'Sport' in climb_types else False
    trad_mask = (user_preds['type'] == 'Trad') & (user_preds['grade_numeric'].isin(range(min_climb_grade, max_climb_grade+1))) if 'Trad' in climb_types else False                                        
                                                    
    if area_ids == []: 
        area_mask = True                                        
       # return user_preds[(user_preds['type'].isin(climb_types)) & (user_preds['grade_numeric'].isin(range(min_grade, max_grade+1))) & (user_preds['pitches'].isin(list(range(pitches, 100 if pitches != 1 else 2))))].sort_values('prediction', ascending = False).head(n).drop(columns = ['description', 'star_ratings'])
    else:
        subareas = functools.reduce(operator.iconcat, [area_tree.get_children(area_id) for area_id in area_ids], [])
        area_mask = user_preds['area_id'].isin(subareas)
    
    pitch_mask = (user_preds['pitches'].isin(pitch_range))
    mask = (boulder_mask | sport_mask | trad_mask) & area_mask & pitch_mask                                            
                                                  
    return user_preds[mask].sort_values('prediction', ascending = False).head(n).drop(columns = ['description', 'star_ratings'])

def get_similar_users(user_id):
    pass

def get_similar_climbs(route_id, k = 10 ):
    
    route_inner_id = knn.trainset.to_inner_iid(route_id)
    route_neighbors = knn.get_neighbors(route_inner_id, k=k)
    route_neighbors = (knn.trainset.to_raw_iid(inner_id)
                       for inner_id in route_neighbors)
    
    return routes[routes['route_id'].isin(route_neighbors)].drop(columns = ['star_ratings', 'description'])


climb_grades = pd.read_csv('../data/climb_grades.csv')
climb_grades = climb_grades[climb_grades['grade'] != '5.?']



boulder_grades = pd.read_csv('../data/boulder_grades.csv')
boulder_grades = boulder_grades[boulder_grades['grade'] != 'V?']



grade_dict = {v:k for k,v in boulder_grades.to_dict()['grade'].items()}
grade_dict.update({v:k for k,v in climb_grades.to_dict()['grade'].items()})



routes = pd.read_csv('../data/routes.csv')
areas = pd.read_csv('../data/areas.csv')
ratings = pd.read_csv('../data/ratings.csv')
svd = pickle.load(open('../pickle/svd.pkl', 'rb'))




area_tree = MPAreaTree(areas)
st.image('mplogo.png', width = 50)
st.title('mountainproject.com Rock Climbing Recommender System')


def get_area_descriptions(area_ids):
	if area_ids == []:
		return 'All Areas'
	else:
		return ', '.join([area_tree.get_area_name(area_id) for area_id in area_ids])


user_id = int(st.sidebar.text_input('mountainproject.com user ID:', value = 200503731))
area_id = st.sidebar.text_input('mountainproject.com area ID:', value = 105819641)
if area_id == '':
	area_ids = []
else:
	area_ids = [int(area) for area in area_id.split(',')]

st.sidebar.write(get_area_descriptions(area_ids))
#st.write(rec_type)

climb_types = st.sidebar.multiselect('Climb Type(s)', options = ['Sport', 'Trad', 'Boulder'], default = ['Sport'])



if 'Sport' in climb_types or 'Trad' in climb_types:
	climb_grade_range = st.sidebar.select_slider(
		'Min/Max Climb Grade',
		options = list(climb_grades['grade'].values), value = ('5.10', '5.13'),
		key = 'climbgrade')
else:
	climb_grade_range = ['3rd', '5.15d']
if 'Boulder' in climb_types:
	boulder_grade_range = st.sidebar.select_slider(
		'Min/Max Boulder Grade',
		options = list(boulder_grades['grade'].values), value = ('V0', 'V10'),
		key = 'bouldergrade')
else:
	boulder_grade_range = ['V-easy','V17']
#st.write(min_climb_grade)

#boulder_cb = st.checkbox('Boulder')
#sport_cb = st.checkbox('Sport')
#trad_cb = st.checkbox('Trad')

#st.write(not boulder_cb and not sport_cb and not trad_cb )

#if climb_type in ['Sport', 'Trad']:
#	min_climb_grade = st.sidebar.selectbox('Minimum Climb Grade', options = climb_grades)
#	max_climb_grade = st.sidebar.selectbox('Maximum Climb Grade', options = climb_grades)
#else:

#	min_boulder_grade = st.sidebar.selectbox('Minimum Boulder Grade', options = boulder_grades)
#	max_boulder_grade = st.sidebar.selectbox('Maximum Boulder Grade', options = boulder_grades)


pitch_options = ['Any', 'Exactly 1', 'At least 2', 'At least 3', 'At least 4', 'At least 5', '6+ pitches']
pitches = st.sidebar.selectbox('Pitches', options = pitch_options)
hit_rec = False

max_results = int(st.sidebar.number_input(
		'Max Number of recommendations',
		min_value = 1, max_value = 50, step = 1, value = 10,
		key = 'maxresults'))

if st.sidebar.button('Get Recommendations'):
	hit_rec = True

	if pitches == 'Any':
		pitch_range = range(1, 100)
	elif pitches == 'Exactly 1':
		pitch_range = range(1,2)
	else:
		pitch_range = range(pitch_options.index(pitches),100)
	
	min_climb_grade = grade_dict[climb_grade_range[0]]
	max_climb_grade = grade_dict[climb_grade_range[1]]
	min_boulder_grade = grade_dict[boulder_grade_range[0]]
	max_boulder_grade = grade_dict[boulder_grade_range[1]]

	if area_id == '':
		area_ids = []
	else:
		area_ids = [int(area) for area in area_id.split(',')]

	rec = get_top_n(user_id, n= max_results, area_ids = area_ids,
	  climb_types = climb_types, min_climb_grade = min_climb_grade,
	  max_climb_grade = max_climb_grade,min_boulder_grade = min_boulder_grade,
	  max_boulder_grade = max_boulder_grade, pitch_range = pitch_range, ignore_completed = True)

	rec['location'] = rec['area_id'].map(area_tree.get_link_chain)
	#rec.apply(lambda x: x['location'] x['location']+ ' > ' + f'<a target="_blank" href="{mp_route_url}{row["route_id"]}">{row["route_name"]}</a>')
	for i, row in rec.iterrows():		
		rec.loc[i,'location'] = row['location'] + ' > ' + f'<a target="_blank" href="{mp_route_url}{row["route_id"]}">{row["route_name"]}</a>'



size = 12
def font_style(x):
	return f'<p style = "font-size:{size}px">{x}</p>'
	#rec = pd.read_csv('./test_rec.csv')
if(hit_rec):
	col1, col2 = st.beta_columns(2)

	st.write(f'Top {max_results} climbing recommendations for user {user_id}')
	
	rec2 = rec[['route_name','type',  'grade',  'pitches', 'score', 'votes','location']].reset_index(drop = True)

	rec2.index += 1
	for column in rec2.columns:
		rec2[column] = rec2[column].map(font_style)
	st.write(rec2.to_html(escape = False), unsafe_allow_html=True)
	#st.table(rec.drop(columns = ['description','star_ratings']))

	st.write('')


	m = folium.Map(
	    location=[41,-95],
	    tiles='Stamen Terrain',
	    zoom_start=4
	)



	for i, row in rec.iterrows():

		lat = areas[areas['area_id'] == row['area_id']].iloc[0]['latitude']
		lon = areas[areas['area_id'] == row['area_id']].iloc[0]['longitude']
		area_name = areas[areas['area_id'] == row['area_id']].iloc[0]['area_name']


		tooltip = f"{row['route_name']} {row['type']} {row['grade']} \n({area_name})"
		folium.Marker([lat,lon], popup = tooltip).add_to(m)
	
			
					#folium.Marker([lat,lon], popup = tooltip).add_to(m)
	
	folium_static(m)
	#st.map(wy_df)
else:
	m = folium.Map(
	    location=[41,-95],
	    tiles='Stamen Terrain',
	    zoom_start=4
	)
	
	for i, row in areas.iterrows():
		if row['parent_id'] == 0:
			lat = row['latitude']
			lon = row['longitude']
			area_name = row['area_name']


			tooltip = f"{area_name}"
			folium.Marker([lat,lon], popup = tooltip).add_to(m)
	folium_static(m)


