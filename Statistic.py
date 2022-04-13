#!/usr/bin/env python
# coding: utf-8



from platform import python_version
print('using python version:' , python_version())

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime


# In[30]:


START_DAY = datetime.date(2015,1,5)
END_DAY = datetime.date(2016,12,31)

FILE_DIR = './preprocessing/'


# In[37]:


class Stock(object):
    '''
    股票資料可視化
    '''
    def __init__(self,file = os.listdir(FILE_DIR),start = START_DAY,end = END_DAY):
        self.file = file
        self.start = start
        self.end = end
    def plot_movement(self):
        plt.figure(figsize = (15,10))
        plt.title('Stock Movement',fontsize = 20)
        for name in self.file:
            df = pd.read_csv(FILE_DIR+'re_data/'+ name ,encoding = 'utf-8',index_col = 'date',parse_dates=['date'])
            df = df.loc[self.start:self.end,:]
            x = df.index
            y = df['close']/df['close'][0]
            plt.plot(x,y*100,linewidth=2.0,label = name[3:7])
        plt.legend(loc='upper right')
        plt.grid(True)
    ##log = True會plot log_return
    def plot_return(self,log = True):
        plt.figure(figsize = (15,10))
        plt.title('Stock Return Movement',fontsize = 20)
        if log:
            df = pd.read_csv(FILE_DIR+'log_return.csv' ,encoding = 'utf-8',index_col = 'date',parse_dates=['date'])
            df = df.loc[self.start:self.end,:]
            x = df.index
            for stock in df.columns:
                plt.plot(x,df[stock] ,linewidth=2.0,label = stock)
            plt.legend(loc='upper right')
            plt.grid(True)
        else:
            df = pd.read_csv(FILE_DIR+'stock_return.csv' ,encoding = 'utf-8',index_col = 'date',parse_dates=['date'])
            df = df.loc[self.start:self.end,:]
            x = df.index
            for stock in df.columns:
                plt.plot(x,df[stock] ,linewidth=2.0,label = stock)
            plt.legend(loc='upper right')
            plt.grid(True)

