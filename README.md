# Silent Viper Investments (SVI) - Trading Tools

## Purpose
This repository was created from my journey through my lessons in [Udacity's](https://www.udacity.com) AI for Trading Nano Degree Program. It is a culmination of the insights I gather to perform stock analysis using Alpha and Beta factors.

The project will consist of some python modules and then example notebooks on how to apply these functions. The current source data for stock prices are retrieved from [Yahoo Finance](https://finance.yahoo.com/).

I have also tied this project to setting buy/sell orders using TD Ameritrade.  To do this you will need is to obtain a software key from [Ameritrade Developer](https://developer.tdameritrade.com/). If you want to analyze your own stock portfolios, and you have an Ameritrade account, then the library has an authentication method and funcitons to pull your data from Ameritrade.



If you follow this project, you will notice large swings in the tools provided. This is because along with some sound strategies for performing stock analysis, Udacity provides the underlying theory prior to a final solution, and since I'm building this as I go along, you will see things disappear because they are replaced by some other higher level concept. 

## Pipeline Limitations
Through the course Udacity promotes the use of [Quantopian's Zipline](https://github.com/quantopian/zipline) modules. When I started using this library, I had a hard time getting the latest stock prices and was only receiving the sample price histories. I struggled with making it work, even trying to create my own bundles from the Ameritrade data, but decide to abandon the effort an use the [Pandas Datareader](https://pandas-datareader.readthedocs.io/en/latest/). More specifically the [Yahoo Daily Reader](https://pandas-datareader.readthedocs.io/en/latest/readers/yahoo.html).

This forced me to create my own Alpha factors toolset using Pandas. I think I did a pretty good job of creating a frame work for this, but down the road I will figure out the zipline stuff so that I can work with a larger number of stocks in my analysis. Right now I limit the stocks to the S&P 500 list, which is obtained using the lastest from [Wikipedia's S&P 500 page](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).

## General Usage (Under Construction)
- Setup:
  - This project uses Ameritrade Developer API and accounts to gather stock and portfolio data
    - You can make necessary changes to use your own source 
  - Get an Ameritrade account [TD Ameritrade](https://www.tdameritrade.com/)
  - Get an Ameritrade developers consumer key  [Ameritrade Developer API](https://developer.tdameritrade.com/)
  - Python 3.8
  - Python modules are listed in the requirements.txt file

### Overview
```mermaid
journey
    title Create and Use Trading Strategy
    section Eval Trading Strategy
      Edit Config: 5: Me
      Clear out files: 5: Me
      Run eval script: 5: Me
      Move final strategies: 5: Me
    section Use Strategies
      Define Accounts: 5: Me
      Define Strategy: 5: Me
      Compute Holdings: 5: Me
      Run Trades: 5: Me
```
```mermaid
stateDiagram-v2
create_and_backtest_stock_selection_strategies.py --> trading_strategy_evaluation/config/strategy_name_final
trading_strategy_evaluation/config/strategy_name_final --> data_directory/strategy_name_final
```


  
# General Flow
```mermaid
stateDiagram-v2
[*] --> Determin_Strategy_to_Use
Determin_Strategy_to_Use --> Move_Stragegy_to_Datadirectory
gather_sp500_price_histories --> price_histories_yahoo.csv
price_histories_yahoo.csv --> generate_alpha_beta_factors
generate_alpha_beta_factors --> all_factors.csv
generate_alpha_beta_factors --> daily_beta.pickle
price_histories_yahoo.csv --> generate_ai_alpha_model
all_factors.csv --> generate_ai_alpha_model
generate_ai_alpha_model --> alpha_ai_model.pickle
alpha_ai_model.pickle --> generate_ai_alpha
generate_ai_alpha --> alpha_vectors.csv
price_histories_yahoo.csv --> portfolio_XXX_adjust_holdings
alpha_vectors.csv --> portfolio_XXX_adjust_holdings
daily_beta.pickle --> portfolio_XXX_adjust_holdings

portfolio_XXX_adjust_holdings --> [*]
```

## Model Evaluation
```mermaid
sequenceDiagram
participant gather_sp500_price_histories
participant generate_alpha_beta_factors
participant generate_ai_alpha_model
participant trading_factors_yahoo
participant nonoverlapping_estimator

gather_sp500_price_histories ->> price_histories_yahoo.csv: Get S&P 500 Price histories

generate_alpha_beta_factors ->> trading_factors_yahoo: Generate Alpha Factors
trading_factors_yahoo --> price_histories_yahoo.csv: Uses Price histories
trading_factors_yahoo -) generate_alpha_beta_factors: Alpha Factors
generate_alpha_beta_factors ->> all_factors.csv: Store Alpha Factors

generate_ai_alpha_model ->> trading_factors_yahoo: Get Factor Return Quantiles
trading_factors_yahoo --> price_histories_yahoo.csv: Uses Price histories
trading_factors_yahoo -) generate_ai_alpha_model: Factor Return Quantiles
generate_ai_alpha_model --> all_factors.csv: Uses Alpha Factors
generate_ai_alpha_model --> generate_ai_alpha_model: Set Model Input Features to Alhpa Factors
generate_ai_alpha_model --> generate_ai_alpha_model: Adjust Factor Return Quantiles by Forward Prediction Days
generate_ai_alpha_model --> generate_ai_alpha_model: Set Model Target Values to Factor Return Quantiles
generate_ai_alpha_model ->> nonoverlapping_estimator: Fit model (Training Data)
nonoverlapping_estimator -) generate_ai_alpha_model: AI Alpha Model
generate_ai_alpha_model ->> alpha_ai_model.pickle: Store Alpha Model
```

## Using model to construct datasets
```mermaid
sequenceDiagram
participant gather_sp500_price_histories
participant generate_alpha_beta_factors
participant generate_ai_alpha
participant trading_factors_yahoo

gather_sp500_price_histories ->> price_histories_yahoo.csv: Store S&P 500 Price histories

generate_alpha_beta_factors ->> trading_factors_yahoo: Generate Alpha Factors
trading_factors_yahoo --> price_histories_yahoo.csv: Uses Price histories
trading_factors_yahoo -) generate_alpha_beta_factors: Alpha Factors
generate_alpha_beta_factors ->> all_factors.csv: Store Alpha Factors

generate_alpha_beta_factors ->> trading_factors_yahoo: Get Factor Returns
trading_factors_yahoo --> price_histories_yahoo.csv: Use Price Histories
trading_factors_yahoo -) generate_alpha_beta_factors: Factor Returns
generate_alpha_beta_factors ->> trading_factors_yahoo: Get Daily Risk Models
trading_factors_yahoo --> trading_factors_yahoo: Use Factor Returns
trading_factors_yahoo -) generate_alpha_beta_factors: Daily Risk Models
generate_alpha_beta_factors ->> daily_beta.pickle: Store Beta Factors

generate_ai_alpha ->> trading_factors_yahoo: Add Alpha Score
trading_factors_yahoo --> all_factors.csv: Use Alpha Factors
trading_factors_yahoo --> alpha_ai_model.pickle: Use AI Alpha Model
trading_factors_yahoo -) generate_ai_alpha: Alpha Scores
generate_ai_alpha --> generate_ai_alpha: Get Alpha Vector
generate_ai_alpha ->> alpha_vectors.csv: Store Alpha Vector
```


## Use datasets to adjust portfolio
```mermaid
sequenceDiagram
participant portfolio_XXX_adjust_holdings
participant portfolio_optimizer
participant ameritrade_functions
portfolio_XXX_adjust_holdings ->> portfolio_optimizer: Find optimal holdings

portfolio_optimizer --> price_histories_yahoo.csv: Use Price Histories
portfolio_optimizer --> alpha_vectors.csv: Use Alpha Vector
portfolio_optimizer --> daily_beta.pickle: Use Beta Facotrs
portfolio_optimizer -) portfolio_XXX_adjust_holdings: New portfolio holdings

portfolio_XXX_adjust_holdings ->> ameritrade_functions: Get current holdings
ameritrade_functions -) portfolio_XXX_adjust_holdings: Current holdings
portfolio_XXX_adjust_holdings --> portfolio_XXX_adjust_holdings: Compute new investment amount
portfolio_XXX_adjust_holdings --> portfolio_XXX_adjust_holdings: Determine adjustments
portfolio_XXX_adjust_holdings ->> ameritrade_functions: Place Sell Orders
portfolio_XXX_adjust_holdings ->> ameritrade_functions: Place Buy Orders
```