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
# print(returns.mean() * 252)
# print(returns.cov() * 252)

#Setup random porfolio weights
weights = np.random.random(noa)
weights /= np.sum(weights)

#Derive Porfolio Returns & simulate various 2500x combinations
# print(np.sum(returns.mean() * weights) * 252)
# print(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

portfolio_returns = []
portfolio_volatility = []
for p in range (2500):
    weights = np.random.random(noa) # generate a list of random floats between 0 and 1
    weights /= np.sum(weights)      # generate a list of random floats between 0 and 1 that add to 1
    portfolio_returns.append(np.sum(returns.mean() * weights) * 252)
    portfolio_volatility.append(np.sqrt(np.dot(weights.T,
                                np.dot(returns.cov() * 252, weights))))
portfolio_returns = np.array(portfolio_returns)
portfolio_volatility = np.array(portfolio_volatility)

plt.figure(figsize = (8, 4))
plt.scatter(portfolio_volatility, portfolio_returns, c = portfolio_returns / portfolio_volatility, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
plt.show()

def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(returns.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

# cons: np.sum(x) - 1 is the function defining the constraint
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa)) # tuple of 10 tuples (0, 1)
noa * [1. / noa,]

optimial_sharpe_ratio = sco.minimize(min_func_sharpe, noa * [1. / noa,], method = 'SLSQP',
                    bounds = bnds, constraints = cons)


print("\n***Maximization of Sharpe Ratio***")

print(optimial_sharpe_ratio['x'].round(3))
print(statistics(optimial_sharpe_ratio['x']).round(3))

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optimal_variance = sco.minimize(min_func_variance, noa * [1. / noa,], method = 'SLSQP',
                    bounds = bnds, constraints = cons)
print("\n***Minimizing Variance***")
#print(optimal_variance)
print(optimal_variance['x'].round(3))
print(statistics(optimal_variance['x']).round(3))

def min_func_portfolio(weights):
    return statistics(weights)[1]



bnds = tuple((0, 1) for x in weights)

target_returns = np.linspace(0.0, 0.25, 50)
target_volatilities = []
for tret in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_portfolio, noa * [1. / noa,], method = 'SLSQP',
                       bounds = bnds, constraints = cons)
    target_volatilities.append(res['fun'])
target_volatilities = np.array(target_volatilities)

########################## Efficient Frontier graph ###########################

plt.figure(figsize = (8, 4))
plt.scatter(portfolio_volatility, portfolio_returns,
            c = portfolio_returns / portfolio_volatility, marker = 'o')   
# random portfolio composition
plt.scatter(target_volatilities, target_returns,
            c = target_returns / target_volatilities, marker = 'x') # efficient frontier values
# efficient frontier
plt.plot(statistics(optimial_sharpe_ratio['x'])[1], statistics(optimial_sharpe_ratio['x'])[0],
         'r*', markersize = 15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optimal_variance['x'])[1], statistics(optimal_variance['x'])[0],
         'y*', markersize = 15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe ratio')
plt.show()

###############################################################################

ind = np.argmin(target_volatilities)        # returns index of smallest element  
efficient_volatilities = target_volatilities[ind:]       # takes values greater than the min variance
efficient_returns = target_returns[ind:]                                    ####
tck = sci.splrep(efficient_volatilities, efficient_returns)     # BSpline object representation
# tck is a tuple (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.

def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der = 0)         # evaluate a BSpline
def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der = 1)         # evaluate a BSpline with first derivation

def equations(p, rf = 0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])


print(opt)
print(np.round(equations(opt), 6))

########################## Zoomed out graph with CML ##########################

plt.figure(figsize = (8, 4))
plt.scatter(portfolio_volatility, portfolio_returns,
            c = (portfolio_returns - 0.01) / portfolio_volatility, marker = 'o')
# random portfolio composition
plt.plot(efficient_volatilities, efficient_returns, 'g', lw = 4.0)
# efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw = 1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize = 15.0)
plt.grid(True)
plt.axhline(0, color = 'k', ls = '--', lw = 2.0)
plt.axvline(0, color = 'k', ls = '--', lw = 2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

###############################################################################

cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
# noa * [1. / noa,] means an array of noa values each with 1/noa 
res = sco.minimize(min_func_portfolio, noa * [1. / noa,], method = 'SLSQP',
                   bounds = bnds, constraints = cons)

print("\n***Optimal Tangent Portfolio***")
print(res['x'].round(3))

print("\n***Optimal Tangent Portfolio Expected Return***")
print(f(opt[2]).round(3))
print("\n***Optimal Tangent Portfolio Expected Volatility***")
print(opt[2].round(3))

opt_eret_evol_df = pd.DataFrame([f(opt[2]), opt[2]], columns = ['Value'], index = ['E(r)', 'sigma'])
print(opt_eret_evol_df)

opt_eret_evol_df.to_excel('Optimal Tangent Portfolio Values.xlsx', sheet_name = "OPT Values", index = True)

opt_df = pd.DataFrame.from_dict(res['x'])
opt_df = opt_df.rename(columns = {0: "Weights"}, 
                       index = {0: "BHP", 1: "CSL", 2: "RIO", 3: "CBA", 4: "WOW", 5: "WES", 6: "TLS", 7: "AMC", 8: "BXB", 9: "FPH"})
opt_df = opt_df.round(3)

opt_df.to_excel('Optimal Tangent Portfolio.xlsx', sheet_name = "Optimal Tangent Portfolio", index = True)

