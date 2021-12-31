# -*- coding: utf-8 -*-
"""
Created on Fri Apr 7 15:28:19 2020

@author: Rajarshi
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib import rcParams 
import seaborn as sb
import scipy
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

%matplotlib inline

rcParams['figure.figsize']= 15,8
sb.set_style('whitegrid')


data=pd.read_csv("time-of-day-per-month-ROADSIDE.csv")
data.head()

newdata=pd.read_csv("ReduceData_Roadside.csv")
newdata.head()


data.fillna(0, inplace=True)
newdata.fillna(0, inplace=True)

#Function to calculate ROADSIDE SO2 individual pollutant index(ss)
def calculate_ss(RSO2):
    ss=0
    if (RSO2<=40):
     ss= RSO2*(50/40)
    if (RSO2>40 and RSO2<=80):
     ss= 50+(RSO2-40)*(50/40)
    if (RSO2>80 and RSO2<=380):
     ss= 100+(RSO2-80)*(100/300)
    if (RSO2>380 and RSO2<=800):
     ss= 200+(RSO2-380)*(100/800)
    if (RSO2>800 and RSO2<=1600):
     ss= 300+(RSO2-800)*(100/800)
    if (RSO2>1600):
     ss= 400+(RSO2-1600)*(100/800)
    return ss


#here it's calculated on redundent data.
data['ss']=data['RSO2'].apply(calculate_ss)
newdata['ss']=newdata['RSO2'].apply(calculate_ss)
df= data[['RSO2','ss']]
ndf=newdata[['RSO2','ss']]

df.head()
ndf.head()

#Function to calculate RNO2 individual pollutant index(ni)
def calculate_ni(RNO2):
    ni=0
    if(RNO2<=40):
     ni= RNO2*50/40
    elif(RNO2>40 and RNO2<=80):
     ni= 50+(RNO2-14)*(50/40)
    elif(RNO2>80 and RNO2<=180):
     ni= 100+(RNO2-80)*(100/100)
    elif(RNO2>180 and RNO2<=280):
     ni= 200+(RNO2-180)*(100/100)
    elif(RNO2>280 and RNO2<=400):
     ni= 300+(RNO2-280)*(100/120)
    else:
     ni= 400+(RNO2-400)*(100/120)
    return ni
data['ni']=data['RNO2'].apply(calculate_ni)
df= data[['RNO2','ni']]
df.head()

#Function to calculate RPM2.5 individual pollutant index(rpi)
def calculate_rpi(RPM25):
    rpi=0
    if(RPM25<=30):
     rpi=RPM25*50/30
    elif(RPM25>30 and RPM25<=60):
     rpi=50+(RPM25-30)*50/30
    elif(RPM25>60 and RPM25<=90):
     rpi=100+(RPM25-60)*100/30
    elif(RPM25>90 and RPM25<=120):
     rpi=200+(RPM25 -90)*100/30
    elif(RPM25>120 and RPM25<=250):
     rpi=300+(RPM25-120)*(100/130)
    else:
     rpi=400+(RPM25-250)*(100/130)
    return rpi
data['rpi']=data['RPM25'].apply(calculate_rpi)
df= data[['RPM25','rpi']]
df.tail()

#here it's calculated on redundent data.
newdata['rpi']=newdata['RPM25'].apply(calculate_rpi)
ndf=newdata[['RPM25','rpi']]
ndf.tail()

#Function to calculate RPM10 individual pollutant index(spi)
def calculate_spi(RPM10):
    spi=0
    if(RPM10<=50):
     spi=RPM10
    if(RPM10<50 and RPM10<=100):
     spi=RPM10
    elif(RPM10>100 and RPM10<=250):
     spi= 100+(RPM10-100)*(100/150)
    elif(RPM10>250 and RPM10<=350):
     spi=200+(RPM10-250)
    elif(RPM10>350 and RPM10<=450):
     spi=300+(RPM10-350)*(100/80)
    else:
     spi=400+(RPM10-430)*(100/80)
    return spi
data['spi']=data['RPM10'].apply(calculate_spi)
df= data[['RPM10','spi']]
df.tail()


#Function to calculate RNO individual pollutant index(na)
def calculate_na(RNO):
    na=0
    if(RNO<=40):
     na= RNO*50/40
    elif(RNO>40 and RNO<=80):
     na= 50+(RNO-14)*(50/40)
    elif(RNO>80 and RNO<=180):
     na= 100+(RNO-80)*(100/100)
    elif(RNO>180 and RNO<=280):
     na= 200+(RNO-180)*(100/100)
    elif(RNO>280 and RNO<=400):
     na= 300+(RNO-280)*(100/120)
    else:
     na= 400+(RNO-400)*(100/120)
    return na
data['na']=data['RNO'].apply(calculate_na)
df= data[['RNO','na']]
df.head()


#Function to calculate RNXO individual pollutant index(nx)
def calculate_nx(RNXO):
    nx=0
    if(RNXO<=40):
     nx= RNXO*50/40
    elif(RNXO>40 and RNXO<=80):
     nx= 50+(RNXO-14)*(50/40)
    elif(RNXO>80 and RNXO<=180):
     nx= 100+(RNXO-80)*(100/100)
    elif(RNXO>180 and RNXO<=280):
     nx= 200+(RNXO-180)*(100/100)
    elif(RNXO>280 and RNXO<=400):
     nx= 300+(RNXO-280)*(100/120)
    else:
     nx= 400+(RNXO-400)*(100/120)
    return nx
data['nx']=data['RNXO'].apply(calculate_nx)
df= data[['RNXO','nx']]
df.head()

#here it's calculated on redundent data.
newdata['nx']=newdata['RNXO'].apply(calculate_nx)
ndf=newdata[['RNXO','nx']]
ndf.head()

#Function to calculate RO3 individual pollutant index(oi)
def calculate_oi(RO3):
    oi=0
    if(RO3<=40):
     oi= RO3*50/40
    elif(RO3>40 and RO3<=80):
     oi= 50+(RO3-14)*(50/40)
    elif(RO3>80 and RO3<=180):
     oi= 100+(RO3-80)*(100/100)
    elif(RO3>180 and RO3<=280):
     oi= 200+(RO3-180)*(100/100)
    elif(RO3>280 and RO3<=400):
     oi= 300+(RO3-280)*(100/120)
    else:
     oi= 400+(RO3-400)*(100/120)
    return oi
data['oi']=data['RO3'].apply(calculate_oi)
df= data[['RO3','oi']]
df.head()


#here it's calculated on redundent data.
newdata['oi']=newdata['RO3'].apply(calculate_oi)
ndf=newdata[['RO3','oi']]
ndf.head()

#function to calculate the air quality index (AQI) of every data value
#its is calculated as per indian govt standards

def calculate_aqi(ss,ni,spi,rpi,na,nx,oi):
    aqi=0
    if(ss>ni and ss>spi and ss>rpi and ss>na and ss>nx and ss>oi):
     aqi=ss
    if(spi>ss and spi>ni and spi>rpi and spi>na and spi>nx and spi>oi):
     aqi=spi
    if(ni>ss and ni>spi and ni>rpi and ni>na and ni>nx and ni>oi):
     aqi=ni
    if(rpi>ss and rpi>ni and rpi>spi and rpi>na and rpi>nx and rpi>oi):
     aqi=rpi
    if(na>ss and na>ni and na>spi and na>rpi and na>nx and na>oi):
     aqi=na
    if(nx>ss and nx>ni and nx>spi and nx>rpi and nx>na and nx>oi):
     aqi=nx
    if(oi>ss and oi>ni and oi>spi and oi>na and oi>nx and oi>rpi):
     aqi=oi

    return aqi

data['AQI']=data.apply(lambda x:calculate_aqi(x['ss'],x['ni'],x['spi'],x['rpi'], x['na'], x['nx'], x['oi']),axis=1)
df= data[['Month','GMT','ss','ni','rpi','spi','na','nx','oi','AQI']]
df.head()

#function to calculate the new air quality index (AQI) of every data value
#its is calculated as per indian govt standards

def calculate_naqi(ss,rpi,nx,oi):
    naqi=0
    if(ss>rpi and ss>nx and ss>oi):
     naqi=ss
    if(rpi>ss and rpi>nx and rpi>oi):
     naqi=rpi
    if(nx>ss and nx>rpi and nx>oi):
     naqi=nx
    if(oi>ss and oi>nx and oi>rpi):
     naqi=oi

    return naqi

newdata['AQI']=newdata.apply(lambda x:calculate_naqi(x['ss'],x['rpi'], x['nx'], x['oi']),axis=1)
ndf= newdata[['Month','GMT','ss','rpi','nx','oi','AQI']]
ndf.head()
#The Pearson correlation
# =============================================================================
# isolate every polutant
# =============================================================================
# RO3 = data['RO3']
# RNXO = data['RNXO']
# RNO = data['RNO']
# RPM10 = data['RPM10']
# RPM25 = data['RPM25']
# RNO2 = data['RNO2']
# RSO2 = data['RSO2']
# =============================================================================
# =============================================================================


PSS=data[['RNO','RNO2','RNXO','RO3','RPM10','RPM25','RSO2']]
sb.pairplot(PSS)
corr = PSS.corr()
corr

PS=newdata[['RNXO','RO3','RPM25','RSO2']]
sb.pairplot(PS)
corr1 = PS.corr()
corr1

# =============================================================================
# #Graph plotting of AQI 
# =============================================================================

gp=data['AQI']
ngp=newdata['AQI']
Mn=data['Month']
plt.plot(Mn,gp,label ="RAW DATA AQI")
plt.plot(Mn,ngp,label ="REDUNDENT DATA AQI")

plt.xlabel('Month')
plt.ylabel('AQI')
plt.legend(loc='upper right')
plt.title('AQI vs Redundent AQI')
plt.show()


#dynamic time wraping.....


x = newdata['AQI']
y = data['AQI']
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)
plt.plot(path)








