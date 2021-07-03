# Silent Viper Investments (SVI) - Trading Tools
# Pourpose
This repository is being created/morphed as I journey through my lessons in [Udacity's](https://www.udacity.com) AI for Trading Nano Degree Program. As I learn new concepts or techniqes I will build up a libray that someone can use to perform  high level portfolio/stock analysis. 

The project will consist of some python modules and then example notebooks on how to apply these functions. The source data for stock prices are retreived from [Ameritrade Developer API](https://developer.tdameritrade.com/). If you are just doing analyis on a list of tickers, then all you will need is to obtain a software key from Ameritrade Developer. If you want to analyze your own stock portfolios, and you have an Ameritrade account, then the library has an authentication method and funcitons to pull your data from Ameritrade.

# Flow
If you follow this project, you will notice large swings in the tools provided. This is because along with some sound strategies for performing stock analysis, Udacity provides the underlying theory prior to a final solution, and since I'm building this as I go along, you will see things dissapear because they are replaced by some other higher level concept. 

# Pipleline Limitations
Throught the course Udacity promotes the use of [Quantopian's Zipline](https://github.com/quantopian/zipline) modules. When I started using this library, I had a hard time getting the latest stock prices and was only receiving the sample price histories. I struggled with making it work, even trying to create my own bundles from the Ameritrade data, but decide to abandon the effort an use the Ameritrade API to pull down price histories. 

This forced me to create my own Alpha factors tools set using Pandas. I think I did a pretty good job of creating a frame work for this, but down the road I will figure out the zipline stuff so that I can work with a larger number of stocks in my analysis. Right now I limit the stocks to a portfolio and some stocks of interest I have either heard about or they landed in Ameritrade's top 10 movers.

# General Usage (Under Construction)
- Stage 1: Build a stock univers - This notebook is used to determine which stock you will analyze 
  - Start with current portfolio stocks (Ameritrade API)
  - Remove any stocks that should not be a part of the anslysis
  - Add additional stocks you are interested in
  - Add additional stocks using some creativity by doing web searches or some other means
  - Gather 2 years of price histories (Ameritrade API)
  - Store price histories and existing portfolio information
- Stage 2: Generate Alpha
  - Generate Alpha factors
  - Generate standard Alphas
  - Using ML, determine best Alphas (iterative)
  - Using ML, generate Alpha vectors for each day for one year (iterative)
  - Store Alphas (used in backtesting)
  - Store models (used for production or actual analysis)
- Stage 3: Generate Beta
  - Using PCA, generate Beta factors
  - Store daily Beta factors for one year (used for backtesting)
  - Store Beta criteria (used for production or actual analysis)   
- Stage 4: Backtesting
  - Using the pricing history, alphas and betas perform a back test for last one year of data 
  - Determine if models are profitable
- Stage 5: Analyze the lastest data to determine new stock holdings
  - Determine how often you want to adjust your portfolio 
  - Make money! 
  - Start over as necessary

# Files

## portfolio_analysis_ml (Jupter Notebook)

This file puts together an entire flow. From authenticating to TD Ameritrace, getting portfolios and downloading price histories. Once most of this is done, a list of Alpha factors are created and processed using standard ML teqniques.

## utils.py

Various utility functions

## project_helper.py

Will be removed. This came with the classwork. 

## ameritrade_functions.py

Functions to interface with Ameritrade. In order to authenticate, I use Selenium through a Chrome driver. You will need to install a compatable Chrome driver (or one that works with your browser) in order to obtain an short term authentication token from Ameritrade. Once you have this, you can look at your own personal portfolios so that you can work them into your optimized portfolio.

## trading_factors.py

Classes and functions to generate Alpha factors and standard quant factors.

## trading_funcitons.py

Some of these will be moved to utils. This was the starting point for organizing the data and generating information like returns. Some of this functionality (like Returns) will be moved to trading_factors.

## portfolio_optimizer.py

This contains a super class and class for using alpha and beta factors to assemble a portfolio mix.

## Various other notebooks.

These notebooks are at this time outdated, but they were used to construct an optimal portfolio using the optimizer.

# Experimental

# sentiment_finvis (Jupyter Notebook)
Using the finvis website to gain sentiment for various stocks. Only look sback 30 days (I think) and not all the stocks I track can be found there. Eventually we will pull 10K documents (for thos available) to get some idea of sentiment for a ticker. What I found there as well, since some of my stocks are in other countries, that not all of them have these documents.

# table_from_html (Jupyter Notebook)
Another class document used to pull data from web sites

# ameritrade_api (Jupyter Notebook)
A notebook I used to experiment with the ameritrade_functions.py module.
