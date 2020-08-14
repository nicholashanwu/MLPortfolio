import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
import xlsxwriter


client_data = pd.read_excel(r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/Client_Details.xlsx', sheet_name='Data')
client_df = pd.DataFrame(client_data, columns =['risk_profile', 'BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH', 'RF Asset'])
client_df = client_df.drop_duplicates()

print(client_df, end = "\n\n")

optimal_tangent_portfolio_data = pd.read_excel(r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/Optimal Tangent Portfolio.xlsx', sheet_name='Optimal Tangent Portfolio')
optimal_df = pd.DataFrame(optimal_tangent_portfolio_data)
optimal_df = optimal_df.rename(columns = {"Unnamed: 0" : "Equity"})

print(optimal_df, end = "\n\n")

opt_values_data = pd.read_excel(r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/Optimal Tangent Portfolio Values.xlsx', sheet_name='OPT Values')
opt_values_data_df = pd.DataFrame(opt_values_data)
opt_values_data_df = opt_values_data_df.rename(columns = {"Unnamed: 0" : "Variable"})

print(opt_values_data_df, end = "\n\n")

sentiment_data = pd.read_excel(r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/Sentiments.xlsx', sheet_name='Sentiments')
sentiment_data_df = pd.DataFrame(sentiment_data, index = None, columns = None)
sentiment_data_df = sentiment_data_df.rename(columns = {"Unnamed: 0" : "Equity"})
sentiment_data_df = sentiment_data_df.sort_values(by=['% positive'])

under_mean_list = []
above_mean_list = []
positive_mean = round(sentiment_data_df['% positive'].mean(),3)
print(sentiment_data_df.iat[0, 4])

for i in range(1, len(sentiment_data_df)):
    if sentiment_data_df.iat[i, 4] < positive_mean:
        under_mean_list.append(sentiment_data_df.iat[i, 0])
    else:
        above_mean_list.append(sentiment_data_df.iat[i, 0])
        
print(under_mean_list)
print(above_mean_list)

# print(optimal_df.iat[y,x])
print(optimal_df.at[8, 'Weights'])

for i in range(min(len(under_mean_list), len(above_mean_list))):
    
    
    if optimal_df.iat[under_mean_list[i], 0] == 0:
        pass
    else:
        optimal_df.iat[under_mean_list[i], 0] -= 0.05
        optimal_df.iat[above_mean_list[i], 0] += 0.05

print(optimal_df, end = "\n\n")


print(sentiment_data_df, end = "\n\n")


# sentiment_mean = sentiment_data_df.iloc[1].mean()
print("% positive mean: ", positive_mean, "%")


expected_return = opt_values_data_df.iloc[0][1]
expected_volatility = opt_values_data_df.iloc[1][1]
rf = 0.01


for r_profile in range(len(client_df)):
    # print(client_df.iloc[r_profile][11])

    y = (expected_return - rf) / ((10 - client_df.iloc[r_profile][0]) * expected_volatility ** 2)
    if y > 1:
        y = 1
        
    client_df.iat[r_profile, 11] = round(1 - y, 3)
    # print(y)
    
print(client_df)



#3.995
    

# bnds = ((0,0.286), 
#         (0,0.476), 
#         (0,0.292), 
#         (0,0.300), 
#         (0,0.733), 
#         (0,0.600), 
#         (0,0.533), 
#         (0,0.308), 
#         (0,0.167), 
#         (0,0.300))
