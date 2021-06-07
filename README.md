# svi-trading-tools
# Pourpose
This repository is being created/morphed as I journey through my lessons in [Udacity's](https://www.udacity.com) AI for Trading Nano Degree Program. As I learn new concepts or techniqes I will build up a libray that someone can use to perform  high level portfolio/stock analysis. 

The project will consist of some python modules and then example notebooks on how to apply these functions. The source data for stock prices are retreived from [Ameritrade Developer API](https://developer.tdameritrade.com/). If you are just doing analyis on a list of tickers, then all you will need is to obtain a software key from Ameritrade Developer. If you want to analyze your own stock portfolios, and you have an Ameritrade account, then the library has an authentication method and funcitons to pull your data from Ameritrade.

# Flow
If you follow this project, you will notice large swings in the tools provided. This is because along with some sound strategies for performing stock analysis, Udacity provides the underlying theory prior to a final solution, and since I'm building this as I go along, you will see things dissapear because they are replaced by some other higher level concept. 

# Pipleline Limitations
Throught the course Udacity promotes the use of [Quantopian's Zipline](https://github.com/quantopian/zipline) modules. When I started using this library, I had a hard time getting the latest stock prices and was only receiving the sample price histories. I struggled with making it work, even trying to create my own bundles from the Ameritrade data, but decide to abandon the effort an use the Ameritrade API to pull down price histories. 

This forced me to create my own Alpha factors tools set using Pandas. I think I did a pretty good job of creating a frame work for this, but down the road I will figure out the zipline stuff so that I can work with a larger number of stocks in my analysis. Right now I limit the stocks to a portfolio and some stocks of interest I have either heard about or they landed in Ameritrade's top 10 movers.
