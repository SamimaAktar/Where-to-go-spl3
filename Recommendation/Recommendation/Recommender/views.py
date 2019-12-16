from django.shortcuts import render
from django.http import HttpResponse
from .models import user_info
from .Recommender import Recommender
from sklearn.externals import joblib
import pickle
import csv
import sys
import reverse_geocoder as rg 
import pprint
import geocoder
from math import cos, asin, sqrt
from collections import defaultdict
from geopy.geocoders import Nominatim
import googlemaps
def getPlaceName(coordinate):
  result = rg.search(coordinate) 
  # result is a list containing ordered dictionary. 
  pprint.pprint(result)
  return result

def getDistance(dest_coor):
  print(dest_coor)
  lat1,lon1=23.7593572,90.3788136
  lat2,lon2=dest_coor
  p = 0.017453292519943295     #Pi/180
  a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
  return 12742 * asin(sqrt(a))

def getTravelTime(distance,travel_speed):

    return distance/travel_speed

def getName(coordinate):
  geolocator = Nominatim(user_agent="wheretogo")
  location = geolocator.reverse(coordinate)
  return location.address
def predict(id):
  with open('assets/mysite/images/recommender_model.pkl','rb') as file:        
      sys.modules['Recommender']=Recommender
      model=joblib.load(file)
      place_id=model.predict(id)
      print(place_id)
      location=model.getLocationInfo()
      all_place_info=list()
      index=0
      for pid in place_id:
        print("Myid: ", pid)
        #name=getName(location[index])
        name=getPlaceName(location[pid])
        #x=name.split(',')
        single_place_info=defaultdict(list)
        distance=getDistance(location[pid])
        travel_time=getTravelTime(distance,40)
        lat,lng=location[pid]
        single_place_info['lat']=lat
        single_place_info['lng']=lng
        single_place_info['place_id']=pid
        single_place_info['place_name']=name[0]['name']
        single_place_info['country']=name[0]['cc']
        #single_place_info['place_name']=str(x[0])+","+str(x[1])+","+str(x[2])
        #single_place_info['country']=x[len(x)-1]
        single_place_info['distance']=distance
        single_place_info['travel_time']=travel_time
        all_place_info.append(single_place_info)
        index+=1
        #print("Distance:",getDistance(location[0]),"KM","Travel Time:",travel_time,"hours")
        #print(all_place_info)
      user={"all_place_info":all_place_info,
            "coors":location}
      return user
         
  

def home_view(request,*arg,**kwargs):

    user=predict(80)
    return render(request,'home.html',user)

def recommendation(request,*arg,**kwargs):
    id=int(request.POST['id'])
    user=predict(id)
    return render(request,'home.html',user)

def sign_in(request,*arg,**kwargs):

  return render(request,'sign_in.html')
def sign_up(request,*arg,**kwargs):
  
  return render(request,'sign_up.html')


