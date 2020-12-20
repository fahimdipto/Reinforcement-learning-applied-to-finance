import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from sklearn.model_selection import train_test_split

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/AAPL.csv')


df_train = df.sort_values('Date')

env_train = DummyVecEnv([lambda: StockTradingEnv(df_train)])


model = PPO2(MlpPolicy, env_train, verbose=1,tensorboard_log="./PPO2_tensorboard/")
# model.learn(total_timesteps=100000)
# model.save("latest.pkl")
model.load("PPO2_trade_2(1000000).pkl")


obs = env_train.reset()
balance_plot = list()
share_plot = list()
profit_plot = list()

obs_plot = list()
n = 30
for i in range(n):
    action, _states = model.predict(obs)
    obs, rewards, done, info = df_train.step(action)
    obs_plot.append(obs)

    balance, profit, share = df_train.render()
    balance_plot.append(balance)
    share_plot.append(share)
    profit_plot.append(profit)
    if done:
        break

# #Close values
# plt.plot(np.array(obs_plot[3]).ravel(),color='green', linewidth=3)
# plt.show()


#Profit
plt.plot(np.array(profit_plot).ravel(),color='red')
plt.show()

#Buy/Sell
plt.plot(np.array(share_plot).ravel(), linewidth=1)
plt.show()


df_train[:n].plot(x='Date', y=['Open'], figsize=(10,5), grid=True)
plt.show()

# plot_new = list()
# for x in obs_plot:
#     plot_new.append(np.array(x).ravel()[2])
#
# plt.plot(np.array(plot_new).ravel(),color='blue', linewidth=3)
# plt.show()


