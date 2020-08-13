
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

assets = ['BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH']
start_date = '2015-03-30'
end_date = '2020-04-02'
total_days = 1830
# requested_prediction = int(input("Enter the day in the future you would like to predict: "))
requested_prediction = 1

mean_dict = {
    'BHP': 0,
    'CSL': 0,
    'RIO': 0,
    'CBA': 0,
    'WOW': 0,
    'WES': 0,
    'TLS': 0,
    'AMC': 0,
    'BXB': 0,
    'FPH': 0,
    }

def loadDataFromFileDB():
    data = pd.read_excel (r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/ASX200top10.xlsx', sheet_name='Bloomberg raw')
    # dataframe = pd.DataFrame(data, columns = ['Dates', 'BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH'])
    data = pd.DataFrame(data)
    
    return data

def print_volatilities(data, assets):
    
    
    print(np.std(data['BHP']))

def main():
    #Load Data and calculate returns
    data = loadDataFromFileDB()
    
    print_volatilities(data, assets)
    
    for i in mean_dict:
        mean_dict[i] = round(data[i].mean(), 3)
    
    print(mean_dict)
    
    ##################################
    
    data['Dates'] = pd.to_numeric(data['Dates']) # convert the Excel dates into numbers
    first_date = data.iloc[0, 0]                 # record the value of the first date   
    
    data['Dates'] = data['Dates'] - first_date   # convert all dates to become milliseconds elapsed 
    data['Dates'] = data['Dates']/(60*60*24*1000*1000000)   # convert time elapsed to days elapsed

    # print(data['Dates'])
    
    x_date = data['Dates'].values
    y_bhp = data['BHP'].values
    
    plt.plot(x_date, y_bhp)
    plt.show()

    
    x_date = x_date.reshape(-1,1)
    y_bhp = y_bhp.reshape(-1,1)
    
    x_date_train, x_date_test, y_bhp_train, y_bhp_test = train_test_split(x_date, y_bhp, test_size = 1/3, random_state = 0)
    
    regressor = LinearRegression()
    regressor.fit(x_date_train, y_bhp_train)
    
    ################## PLOT TRAINING AND TEST DATA
    
    # viz_train = plt
    # viz_train.scatter(x_date_train, y_bhp_train, color='red', marker='.', s=5)
    # viz_train.plot(x_date_train, regressor.predict(x_date_train), color='blue')
    # viz_train.title('Stock Price vs Date (Training set)')
    # viz_train.xlabel('Days Elapsed')
    # viz_train.ylabel('Stock Price ($USD)')
    # viz_train.show()
    
    # viz_test = plt
    # viz_test.scatter(x_date_test, y_bhp_test, color='red', marker='.', s=5)
    # viz_test.plot(x_date_train, regressor.predict(x_date_train), color='blue')
    # viz_test.title('Stock Price vs Date (Test set)')
    # viz_test.xlabel('Days Elapsed')
    # viz_test.ylabel('Stock Price ($USD)')
    # viz_test.show()
    
    ##################
     
    # x_date_train = x_date_train + 1830
    # print(x_date_train)
    
    y_pred = regressor.predict(np.array([requested_prediction + 1830]).reshape(-1, 1))
    
    
    date_array = np.array([[]])
    for i in range(len(x_date)):
        date_array = np.append(date_array, i + 1830)
        
    date_array = date_array.reshape(-1, 1)
    print(len(x_date_test))
    
    print(y_pred)
    
    viz_test = plt
    viz_test.scatter(date_array, regressor.predict(date_array), color='red', marker='.', s=5)
    viz_test.plot(x_date_train, regressor.predict(x_date_train), color='purple')
    viz_test.title('Stock Price vs Date (Predicted set)')
    viz_test.xlabel('Days Elapsed')
    viz_test.ylabel('Stock Price ($USD)')
    viz_test.show()
    
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    

if __name__ == '__main__':
    main()



