# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:46:58 2017

This code used Good Morning code to get fundamental data from Morning Star for
10 companies specifiying which fundamental analysis ratio's to choose
from the dataframe

@author: Ferhat
"""

import good_morning as gm
import pandas as pd


country = {'USA':'US_Stocks.csv', 'UK':'UK_Stocks.csv', 'JAP':'JAP_Stocks', 'GER':'Germany_Stocks', 'FRA':'FRA_Stocks', 'ITA':'ITA_Stocks','ITA2':'ITA_Stocks2','HK':'HK_Stocks'}
country_code = 'UK'
input_path = 'C:/Users/Ferhat/FinanceCode/Good-Morning/good-morning-master/input/'
output_path = 'C:/Users/Ferhat/FinanceCode/Good-Morning/good-morning-master/output/'
data = pd.read_csv('%s' % input_path + '%s' % country[country_code],encoding = "ISO-8859-1")
df = pd.DataFrame(columns = ['Ticker','Net Income','EPS','Dividends','Payout Ratio',\
                             'Shares','Book Value/Share','Current Ratio','Quick Ratio',\
                             'Free Cashflow/Share','Revenue','Gross Margin','Operating Margin',\
                             'EBT Margin','Net Margin', 'Return on Assets',\
                             'Return on Equity','Return on Invested Capital','Interest Coverage',\
                             'EPS Year on Year', 'Total Assets','Debt/Equity'])


def append_stock(i):
    #global country_code
    global df
    global data
        
    kr = gm.KeyRatiosDownloader()
    Ticker2 = data.loc[i,"Ticker"]
    kr_frames = kr.download(Ticker2)
    kr0=kr_frames[0]
    kr1=kr_frames[1]
    kr2=kr_frames[2]
    kr5=kr_frames[5]
    kr6=kr_frames[6]
    kr8=kr_frames[8]
    kr9=kr_frames[9]
    df=df.append({'Ticker':data.loc[i,"Ticker"],'Company Name':data.loc[i,"Name"],\
                  'Current Ratio':kr9.iloc[0,-1],\
                  'Quick Ratio':kr9.iloc[1,-1],\
                  'EPS':kr0.iloc[5, -1],\
                  'Debt/Equity':kr9.iloc[3,-1],\
                    'Net Income':kr0.iloc[4,-1],\
                    'Dividends':kr0.iloc[6,-1],\
                    'Payout Ratio':kr0.iloc[7,-1],\
                    'Shares':kr0.iloc[8,-1],\
                    'Book Value/Share':kr0.iloc[9,-1],\
                    'Free Cashflow/Share':kr0.iloc[13,-1],\
                    'Revenue':kr1.iloc[0,-1],\
                    'Gross Margin':kr1.iloc[2,-1],\
                    'Operating Margin':kr1.iloc[6,-1],\
                    'EBT Margin':kr1.iloc[8,-1],\
                    'Net Margin':kr2.iloc[1,-1],\
                    'Return on Assets':kr2.iloc[3,-1],\
                    'Return on Equity':kr2.iloc[5,-1],\
                    'Return on Invested Capital':kr2.iloc[6,-1],\
                    'Interest Coverage':kr2.iloc[7,-1],\
                    'EPS Year on Year':kr6.iloc[0,-1],\
                    '5-Year Net Income %':kr5.iloc[2,-2],\
                    'Total Assets':kr8.iloc[8,-1]\
                    },\
                  ignore_index = True)  
    return df

for i in range (0,len(data)):
    try:
        append_stock(i) 
    
    except ValueError:
        try:
            append_stock(i)  
        except ValueError:
            try:
                append_stock(i) 
            except ValueError:
                pass
        
          
#Save all raw data to output csv
df.to_csv('%s' % output_path + '%s_df.csv' % country_code ,na_rep="0")

#Dividend Cover    
df.loc[:,24]=df.iloc[:,3]/df.iloc[:,4]
df.rename(columns={24: 'Dividend Cover'}, inplace=True)

#Set real conditions here, below:
df_condition = df.loc[(df['Debt/Equity'] < 1.0) & (df['Current Ratio'] > 1.0) & (df['5-Year Net Income %'] > 20.0)]

#Save this new conditioned dataframe to csv file
df_condition.to_csv('%s' % output_path + '%s_df_condition.csv' % country_code ,na_rep="0")
    

    