
# This is a class which handles building and extracting information
# from the mountain project area/subarea hierarchy 

# Mountainproject divides different climbing locations into areas and subareas
# an area may contain multiple subareas, which themselves may either contain
# more subareas or a list of routes (but not both)

# Each area is identified by a unique 9-digit ID
# During scraping, we collect each area_id, name, and parent area_id.

# This class converts a pandas DataFrame containing area_ids and parent_ids
# into a dictionary tree structure which can be traversed 
# and queried to obtain information such as 
# - area names
# - list (chain) of parent areas ids/names
# - list (chain) of child areas ids/names
import functools
import operator

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
    
    