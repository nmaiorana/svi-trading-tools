# Creating and Evaluating Strategies

In this section you can generate various model training and backtesting strategies to suite your trading style.

You start in the config folder and define your strategy parameters, and which strategies you want to work with. With the acceptance criteria you give, like minimum viable portfolio return (min_viable_port_return), if the strategy passes it will be moved to the final folder name you provide.

Accepted strategies are automatically skipped, so you can focus on new ones. The idea is that accepted strategies are copied into the super config folder where all the trading decisions are made.

As you move through each evaluation routine, the system saves off files to save time on replaying unnecessary steps. For instance, if you are using 10 years of price histories, you don't need to keep pulling them down everytime you run an evaluation. If you do want to get fresh data, simply remove the file from the data directory and the process will automatically create a fresh copy.

Remember to remove other files down the chain in order of the new data to be used. For instance if you create new alpa-factors, make sure to remove subsequent files which use alpha-factors to construct the next artifact in the chain, like building a new AI alpha model. The chain of artifacts looks like this:

- data/price_histories_yahoo.parquet
- data/all_factors.parquet
- data/strategy_being_evaluated/alpha_ai_model.pickle
- data/strategy_being_evaluated/ai_alpha_factor.parquet
- data/strategy_being_evaluated/alpha_vectors.parquet
- data/strategy_being_evaluated/daily_betas.pickle
- data/final_strategy_name/* (this gets moved to production use if desired)

## TODO: Add description of config file options

