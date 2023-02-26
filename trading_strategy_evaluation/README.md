# Creating and Evaluating Strategies

In this section you can generate various model training and backtesting strategies to suite your trading style.

You start in the config folder and define your strategy parameters, and which strategies you want to work with. With the acceptance criteria you give, like minimum viable portfolio return (min_viable_port_return), if the strategy passes it will be moved to the final folder name you provide.

Accepted strategies are automatically skipped, so you can focus on new ones. The idea is that accepted strategies are copied into the super config folder where all the trading decisions are made.

As you move through each evaluation routine, the system saves off files to save time on replaying unnecessary steps. For instance, if you are using 10 years of price histories, you don't need to keep pulling them down everytime you run an evaluation. If you do want to get fresh data, simply remove the file from the data directory and the process will automatically create a fresh copy. The script knows to build downstream data files when upstream ones have been changed.

The chain of artifacts looks like this:

- data/price_histories_yahoo.parquet
- data/all_factors.parquet
- data/strategy_name_eval/alpha_ai_model.pickle
- data/strategy_name_eval/ai_alpha_factor.parquet
- data/strategy_name_eval/alpha_vectors.parquet
- data/strategy_name_eval/daily_betas.pickle
- data/strategy_name_final/* (this gets moved to production use if desired)

## Configuration Options
[EVALUATION]
Strategies = (names of all the strategies to evaluate)

[strategy_name]
name = (may remove)
final_name = (may remove)
min_sharpe_ratio = .85 (for each alpha factor, choose the ones with this sharpe value or higher)
ForwardPredictionDays = 5 (How often are you trading)
PredictionQuantiles = 2 (Number of categories to predict)
RandomForestNTrees = 5000 (How big is the AI classifier)
AIAlphaName = AI_ALPHA (name of the AI Alpha Factor)
NumberOfRiskExposures = 20 (How big to make your Betas)
risk_cap = 0.10 (How risky)
weights_max = 0.15 (Max percentage of portfolio for each stock)
weights_min = 0.00 (Min percentage, negative values would be used for shorts)
min_viable_port_return = 5.0 (for backtesting, what is the minimal return to use)
trading_day=MON (or 0) (Not used yet)
