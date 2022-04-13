#!/usr/bin/env python
# coding: utf-8




from platform import python_version
print(python_version())




import pandas as pd
import numpy as np
import scipy.optimize as opt
import imageio
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy.linalg as LA




## 股票價格數據
RETURN_FILENAME = './preprocessing/log_return.csv'

## 股票市值數據
MV_FILENAME = './preprocessing/market_value.csv'

## 候選股票
stock_list = ['2330','2885','2603','2303','2881']
# Model Parameter
TAU = 0.05    # confidence
rf = np.log(1+0.02)/(365/5) ## risk free return


# Backtest Date
START_DAY = datetime.date(2015,1,5)
END_DAY = datetime.date(2019,12,31)




class BlackLitterman(object):
    '''直接丟一個時期的return進來'''
    def __init__(self ,start_day,TAU = TAU, stock_list = stock_list ,period = 100 , P = np.eye(5) , Q = 0.01*np.ones([5,1])):
        self.stock_list = stock_list
        #print(self.stock_list)
        self.return_filename = RETURN_FILENAME ###
        self.mv_filename = MV_FILENAME ###
        self.start_day = start_day
        self.period = period
        self.tau = TAU ###
        self.P = P
        self.Q = Q
    ## 2015-2019 return
    def get_return(self):
        df = pd.read_csv(self.return_filename ,encoding = 'utf-8',index_col = 'date',parse_dates=['date'])
        data = df[self.stock_list]
        return data.loc[self.start_day - datetime.timedelta(self.period+1):self.start_day-datetime.timedelta(1),:] ###
    ## 2015-2019 market value
    def mkt_value_weight(self):
        df = pd.read_csv(self.mv_filename ,encoding = 'utf-8',index_col = 'date',parse_dates=['date'])
        df = df.loc[self.start_day - datetime.timedelta(self.period+1):self.start_day,:]
        data = df[self.stock_list]
        data = data.mean()
        data = data/data.sum()

        return np.array(data)
    def CAPM_return(self):
        '''
        :param stock_cc_ret: 指定T部分的10只股票收益率数据（维度：T * 10）
        :param mkt_weight: 市場權重（维度：1*10）
        :return: risk aversion : delta、先验预期收益率：implied_ret
        '''
 
        mkt_weight = self.mkt_value_weight()
        # 根据股票收益率计算得到协方差矩阵：mkt_cov
        rts = self.get_return()
        mkt_cov = np.array(rts.cov())
        
        # lambd: implied risk-aversion coefficient（风险厌恶系数）
        delta = ((np.dot(mkt_weight, rts.mean())) - rf) / np.dot(np.dot(mkt_weight, mkt_cov), mkt_weight.T)
        # 计算先验预期收益率：implied_ret
        implied_ret = delta * np.dot(mkt_cov, mkt_weight)
        return implied_ret, delta
    ## covariance of the view
    def Omega(self):
        rts = self.get_return()
        SIGMA = np.array(rts.cov())
        shape = self.P.shape
        return np.dot(self.P,np.dot(self.tau*SIGMA,self.P.T)) * np.eye(shape[0])
    
    def posterior_dist(self):
        rts = self.get_return()
        SIGMA = np.array(rts.cov())
        omega = self.Omega()
        tau = self.tau
        PI ,delta = self.CAPM_return() 
        left_term = LA.inv(LA.inv(tau*SIGMA)+np.dot(self.P.T,np.dot(LA.inv(omega),self.P)))
        right_term = np.expand_dims(np.dot(LA.inv(tau*SIGMA) , PI),axis = 1) +  np.dot(self.P.T,np.dot(LA.inv(omega),self.Q))
#         print(np.dot(self.P.T,np.dot(LA.inv(omega),self.Q)))
        post_return = np.dot(left_term,right_term)
        post_sigma = LA.inv(LA.inv(tau*SIGMA) + np.dot(self.P.T,np.dot(LA.inv(omega),self.P)))
        return [post_return,SIGMA]#[post_return,post_sigma]
    def utility_weight(self):
        ret,delta = self.CAPM_return()
        rts = self.get_return()
        SIGMA = np.array(rts.cov())
        post_ret,post_cov = self.posterior_dist()
        weight = np.dot(LA.inv(delta * SIGMA), post_ret)
        return weight


# In[81]:


class Optimization(object):
    def __init__(self, pos_mu , pos_var, stock_list = stock_list):
        self.mu = pos_mu
        self.var = pos_var 
        self.stock_list = stock_list
        self.init_weight = np.array([1 for k in range(len(self.stock_list))])/len(self.stock_list)
    def obj_func(self,weight):
        protf_return = np.dot(weight,self.mu)
        protf_var = np.dot(np.dot(weight,self.var),weight.T)
        sharpe = (protf_return - rf) / (protf_var)**(1/2)
        return np.array([protf_return,protf_var,sharpe])
    def max_sharpe(self,weight):
        return -self.obj_func(weight)[2]
    def best_sharpe(self,print_process = True):
        num_stock = len(self.stock_list)
        bnds = list((0., 1.) for x in range(num_stock))  ### 權重介於0-1之間，有num_stock比權重
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) ### x1+x2+.....+xn = 1
        initial_cond = self.init_weight  ###規劃求解都要給一個起始條件，這邊隨機給
        if print_process:
            print("------------------Initial Condition----------------------") ##只是想讓格式比較好看
            for i,j in enumerate(self.stock_list):
                 print(j," = ",initial_cond[i].round(3))
            print("---------------------------------------------------------\n\n") ##只是想讓格式比較好看
            ## 以下為求解
            solvs = opt.minimize(self.max_sharpe , initial_cond, method='SLSQP',  bounds=bnds, constraints=cons)
            ## SLSQP是一種求解演算法，會用到Jacobian所以求解速度有點慢
            print(solvs)
            print("\n\n------------------Optimal Condition----------------------") ##只是想讓格式比較好看
            for i,j in enumerate(self.stock_list):
                 print(j," = ",solvs['x'].round(4)[i])
        else:
            solvs = opt.minimize(self.max_sharpe , initial_cond, method='SLSQP',  bounds=bnds, constraints=cons)
    
        return solvs['x'].round(4) ##return最高sharpe點的收益風險sharpe

class Optimization_MVO(object):
    def __init__(self, mu , var, stock_list = stock_list):
        self.mu = mu
        self.var = var 
        self.stock_list = stock_list
        self.init_weight = np.array([1 for k in range(len(self.stock_list))])/len(self.stock_list)
    def obj_func(self,weight):
        protf_return = np.dot(weight,self.mu)
        protf_var = np.dot(np.dot(weight,self.var),weight.T)
        sharpe = (protf_return - rf) / (protf_var)**(1/2)
        return np.array([protf_return,protf_var,sharpe])
    def max_sharpe(self,weight):
        return -self.obj_func(weight)[2]
    def best_sharpe(self,print_process = True):
        num_stock = len(self.stock_list)
        bnds = list((0., 1.) for x in range(num_stock))  ### 權重介於0-1之間，有num_stock比權重
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) ### x1+x2+.....+xn = 1
        initial_cond = self.init_weight  ###規劃求解都要給一個起始條件，這邊隨機給
        if print_process:
            print("------------------Initial Condition----------------------") ##只是想讓格式比較好看
            for i,j in enumerate(self.stock_list):
                 print(j," = ",initial_cond[i].round(3))
            print("---------------------------------------------------------\n\n") ##只是想讓格式比較好看
            ## 以下為求解
            solvs = opt.minimize(self.max_sharpe , initial_cond, method='SLSQP',  bounds=bnds, constraints=cons)
            ## SLSQP是一種求解演算法，會用到Jacobian所以求解速度有點慢
            print(solvs)
            print("\n\n------------------Optimal Condition----------------------") ##只是想讓格式比較好看
            for i,j in enumerate(self.stock_list):
                 print(j," = ",solvs['x'].round(4)[i])
        else:
            solvs = opt.minimize(self.max_sharpe , initial_cond, method='SLSQP',  bounds=bnds, constraints=cons)
    
        return solvs['x'].round(4) ##return最高sharpe點的收益風險sharpe
        