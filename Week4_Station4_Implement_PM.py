import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
np.random.seed(8)

assets = ['BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH']
start_date = '2015-03-30'
end_date = '2020-04-20'

# number of assets
noa = 10

data = pd.read_excel(r'C:/Users/nic27/OneDrive/Documents/FINS3645/Assignment Data/Option 1/ASX200top10.xlsx', sheet_name='Bloomberg raw')
df = pd.DataFrame(data, columns =['BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH'])

#Calculate returns and covariance matrix
returns = np.log(df / df.shift(1))
print(returns.mean() * 252)
print(returns.cov() * 252)

#Setup random porfolio weights
weights = np.random.random(noa)
weights /= np.sum(weights)

#Derive Porfolio Returns & simulate various 2500x combinations
print(np.sum(returns.mean() * weights) * 252)
print(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

portfolio_returns = []
portfolio_volatility = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(returns.mean() * weights) * 252)
    portfolio_volatility.append(np.sqrt(np.dot(weights.T,
                                np.dot(returns.cov() * 252, weights))))
portfolio_returns = np.array(portfolio_returns)
portfolio_volatility = np.array(portfolio_volatility)

plt.figure(figsize=(8, 4))
plt.scatter(portfolio_volatility, portfolio_returns, c = portfolio_returns / portfolio_volatility, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(returns.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
noa * [1. / noa,]

opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)
print("***Maximization of Sharpe Ratio***")
#print(opts)
print(opts['x'].round(3))
print(statistics(opts['x']).round(3))

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)
print("****Minimizing Variance***")
#print(optv)
print(optv['x'].round(3))
print(statistics(optv['x']).round(3))

def min_func_port(weights):
    return statistics(weights)[1]

bnds = tuple((0, 1) for x in weights)
target_returns = np.linspace(0.0, 0.25, 50)
target_volatilities = []
for tret in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)
    target_volatilities.append(res['fun'])
target_volatilities = np.array(target_volatilities)

plt.figure(figsize=(8, 4))
plt.scatter(portfolio_volatility, portfolio_returns,
            c = portfolio_returns / portfolio_volatility, marker='o')
# random portfolio composition
plt.scatter(target_volatilities, target_returns,
            c = target_returns / target_volatilities, marker='x')
# efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

ind = np.argmin(target_volatilities)
efficient_volatilities = target_volatilities[ind:]
efficient_returns = target_returns[ind:]
tck = sci.splrep(efficient_volatilities, efficient_returns)
def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)
def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
#print(opt)
#print(np.round(equations(opt), 6))

#################################

plt.figure(figsize=(8, 4))
plt.scatter(portfolio_volatility, portfolio_returns,
            c=(portfolio_returns - 0.01) / portfolio_volatility, marker='o')
# random portfolio composition
plt.plot(efficient_volatilities, efficient_returns, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

#################################

cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                   bounds=bnds, constraints=cons)
print("***Optimal Tangent Portfolio***")
print(res['x'].round(3))