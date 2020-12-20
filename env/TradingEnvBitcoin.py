
# In[102]:

import pandas as pd
from tensortrade.exchanges.simulated import SimulatedExchange
import matplotlib.pyplot as plt

df = pd.read_csv("../data/Coinbase_BTCUSD_1h.csv", skiprows=1)
exchange = SimulatedExchange(data_frame=df, base_instrument='USD', pretransform=True)


# In[103]:
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
difference_all = FractionalDifference(difference_order=0.6)

feature_pipeline = FeaturePipeline(steps=[normalize_price, difference_all])

exchange.feature_pipeline = feature_pipeline

from tensortrade.actions import DiscreteActions

action_scheme = DiscreteActions(n_actions=20, instrument='BTC')


from tensortrade.rewards import SimpleProfit

reward_scheme = SimpleProfit()

from tensortrade.environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 feature_pipeline=feature_pipeline,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme)

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5, 'nminibatches': 1 }

from tensortrade.strategies import StableBaselinesTradingStrategy

strategy = StableBaselinesTradingStrategy(environment=environment,
                                          model=model,
                                          policy=policy,
                                          model_kwargs=params)
performance = strategy.run(steps=10000)
strategy.save_agent(path="ppo_btc_1h")
# strategy.restore_agent(path="agents/ppo_btc_1h")

# In[104]:

performance.net_worth.plot()
plt.show()


# In[105]:
performance.balance.plot()
plt.show()


# In[106]:
# import matplotlib.pyplot as plt
# plt.plot(performance['balance'].values[:20].ravel(),color='blue')
# plt.show()

performance["profit"] = performance["balance"]-10000

# In[106]:
import matplotlib.pyplot as plt
plt.plot(performance['net_worth'].values.ravel(),color='blue')
plt.show()