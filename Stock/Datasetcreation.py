import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

api_key='XXXXXXXXXXXXXXXX'

ticker=input("enter the stock:-")

ts=TimeSeries(key=api_key, output_format='pandas')
data, meta_data=ts.get_intraday(symbol=ticker, interval='5min',outputsize='full')
print(data)

i=1
while i==1:
    data, meta_data=ts.get_intraday(symbol=ticker, interval='5min',outputsize='full')
    data.to_excel('/home/saksham/Stock/livestock.xlsx')
    data.to_csv('/home/saksham/Stock/livestock.csv')
    time.sleep(10)
    i+=1
