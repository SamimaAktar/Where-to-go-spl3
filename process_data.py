import numpy as np
import pandas as pd
import csv
from collections import defaultdict,Counter

checkins = pd.read_csv('../gowalla/gowalla_checkins.csv')
#checkins = pd.read_csv('../LRSbasics-master/dataset/original_data.csv',nrows=5)
process_data={}
for x,ch in checkins.iterrows():
    process_data[(ch[0], ch[1])]=1


with open('../gowalla/preprocessed_data.csv', 'w', newline='') as preprocessed_data:
    checkins_writer = csv.writer(preprocessed_data, delimiter='\t')
    for checkin_info in process_data:
        checkin = [checkin_info[0], checkin_info[1], process_data[checkin_info]];
        checkins_writer.writerow(checkin)





print ("heloo")