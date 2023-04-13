# forex-prediction
!pip install colorama
!pip install tradermade
from google.colab import drive
import pandas as pd
import numpy as np
import numpy.ma as ma
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import tradermade as tm
import datetime
drive.mount('/content/drive')
df = pd.read_excel('/content/drive/My Drive/USDCAD_Day_Predict.xlsx')
df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
Features = df[['Price','Open','Vol.']].copy()
Target_01 = df[['High']].copy()
Target_02 = df[['Low']].copy()
from sklearn.model_selection import train_test_split
X = Features
y = Target_01
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=13)

from sklearn.linear_model import LinearRegression
Linear_model_01 = LinearRegression()
Linear_model_01.fit(X_train, y_train)
Linear_model_01.score(X_test, y_test)
predicted_array1 = Linear_model_01.predict(X_test)
Predicted_01 = pd.DataFrame(predicted_array1, columns = ['Predicted_High'])
result1 = pd.merge(Predicted_01, y_test,left_index=True, right_index=True)
from sklearn.model_selection import train_test_split
m = Features
n = Target_02
m_train, m_test, n_train, n_test = train_test_split(m,n,test_size=0.30,random_state=13)
from sklearn.linear_model import LinearRegression
Linear_model_02 = LinearRegression()
Linear_model_02.fit(m_train, n_train)
Linear_model_02.score(m_test, n_test)
predicted_array2 = Linear_model_02.predict(m_test)
Predicted_02 = pd.DataFrame(predicted_array2, columns = ['Predicted_Low'])
result2 = pd.merge(Predicted_02, n_test,left_index=True, right_index=True)
Predicted_array = (predicted_array1+predicted_array2)/2 
Predicted = pd.DataFrame(Predicted_array, columns = ['Predicted'])
price = df[['Price']]
result = pd.merge(price,Predicted,left_index=True, right_index=True)
decision = pd.merge(result1,result2,left_index=True, right_index=True)
Final_df = pd.merge(result,decision,left_index=True, right_index=True)
date = df[['Date']]
final_decition = pd.merge(date,Final_df,left_index=True, right_index=True)
Price = float(1.3593)
Open = float(1.3597)
Vol = float(73.30)
#Linear_model_01.predict([[Price,Open,Vol]])
j = Linear_model_01.predict([[Price,Open,Vol]])
b = Linear_model_02.predict([[Price,Open,Vol]])
l = pd.DataFrame(j, columns = ['projected_high'])
n = pd.DataFrame(b, columns = ['projected_low'])
Predicted_High_nparrayTo_float = ma.masked_array(l.iat[0, 0], mask =[False]).__float__()
Predicted_Low_nparrayTo_float = ma.masked_array(n.iat[0, 0], mask =[False]).__float__()
Predicted_mean = (Predicted_High_nparrayTo_float + Predicted_Low_nparrayTo_float)/2
projected_high_difference = abs(Predicted_High_nparrayTo_float-Price)
projected_low_difference = abs(Predicted_Low_nparrayTo_float-Price)
if Predicted_mean > Price and projected_high_difference > projected_low_difference:
  print(f"{Fore.RED}Up{Style.RESET_ALL} can be Predicted!")
elif Predicted_mean < Price and projected_high_difference < projected_low_difference:
    print(f"{Fore.RED}Down{Style.RESET_ALL} can be Predicted!")
else:
    print(f"{Fore.RED}SORRY{Style.RESET_ALL},No prediction!!!")
    tm.set_rest_api_key("C9gXIyD3rMn0Eg-bpPyu")
x = "2022-10-20"
vol = float(111.30)
#x = input("Please input your targeted date [yyyy-dd-mm] for first 12 days otherwise yyyy-mm-dd : ")
#vol = input("Please input your targeted dated volume : ")
#tm.live(currency='EURUSD,GBPUSD',fields=["bid", "mid", "ask"]) # returns live data - fields is optional
#df1 = tm.timeseries(currency='USDCAD', start="2022-12-31",end=datetime.datetime.today (),interval="daily",fields=["open", "high", "low","close"])
df1 = tm.timeseries(currency='USDCAD', start=x,end=datetime.datetime.today (),interval="daily",fields=["open", "high", "low","close"])
# returns timeseries data for the currency requested interval is daily, hourly, minute - fields is optional
df2 = df1.head(1)
#df2 = df[df['Date']==df.iat[0,0]]
#df2 = df[df['Date']==input("Please input your targeted date [yyyy-dd-mm] for first 12 days otherwise yyyy-mm-dd : ")]
Price_Previous_Closing = float(df2.iat[0,4])
Open_Previous_Starting = float(df2.iat[0,1])                                                 # input function is in str
Vol_Privious = vol
j = Linear_model_01.predict([[Price_Previous_Closing,Open_Previous_Starting,Vol_Privious]]) #Linear_model_01.predict([[92.13,92.8,7027]])
b = Linear_model_02.predict([[Price_Previous_Closing,Open_Previous_Starting,Vol_Privious]])
l = pd.DataFrame(j, columns = ['projected_high'])
n = pd.DataFrame(b, columns = ['projected_low'])
Predicted_High_nparrayTo_float = ma.masked_array(l.iat[0, 0], mask =[False]).__float__()
Predicted_Low_nparrayTo_float = ma.masked_array(n.iat[0, 0], mask =[False]).__float__()
Predicted_mean = (Predicted_High_nparrayTo_float + Predicted_Low_nparrayTo_float)/2
projected_high_difference = abs(Predicted_High_nparrayTo_float-Price_Previous_Closing)
projected_low_difference = abs(Predicted_Low_nparrayTo_float-Price_Previous_Closing)
print('Predicting Price for :',df2.iat[0,0] )  # pd.to_datetime(df['Date']).dt.normalize()
print('Predicting Price for : ',Price_Previous_Closing)
print('Predicted_mean is : ',Predicted_mean)
print('projected_high_difference is : ',projected_high_difference)
print('projected_low_difference is : ',projected_low_differ)
if Predicted_mean > Price_Previous_Closing and projected_high_difference > projected_low_difference :
  print(f"{Fore.RED}Up{Style.RESET_ALL} can be Predicted!")
elif Predicted_mean < Price_Previous_Closing and projected_high_difference < projected_low_difference:
    print(f"{Fore.RED}Down{Style.RESET_ALL} can be Predicted!")
else:
    print(f"{Fore.RED}SORRY{Style.RESET_ALL},You have no predictions!!!")
    final_decition.to_excel(r'/content/drive/My Drive/USDCAD_Day_Predicted_Decision.xlsx', index=False)
 
 #Completed
