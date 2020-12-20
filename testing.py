#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

from env.StockTradingTestingEnv import StockTradingTestingEnv
import pandas as pd
import matplotlib.pyplot as plt
import ta
import tensorflow as tf

df_train = pd.read_csv('data/IBMTRAIN.csv')
df_test = pd.read_csv('data/IBMTEST.csv')

df_train = ta.add_all_ta_features(df_train, "Open", "High", "Low", "Close", "Volume", fillna=True)
df_test = ta.add_all_ta_features(df_test, "Open", "High", "Low", "Close", "Volume", fillna=True)



# In[103]:


df_train.head()


# In[104]:


df_test.head()


# In[105]:


df_train = df_train.dropna()
df_test = df_test.dropna()


# In[106]:


# df_train = df_train.drop("Unnamed: 0", axis=1)

# df_test = df_test.drop("Unnamed: 0", axis=1)


# In[107]:


df_test.describe



# In[108]:


env_train = DummyVecEnv([lambda: StockTradingEnv(df_train)])
env_test = DummyVecEnv([lambda: StockTradingTestingEnv(df_test)])


# In[109]:


df_test.isnull().values.any()


# In[110]:
#
# policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64,64,64,64])
model = PPO2(MlpPolicy, env_train, verbose=1,tensorboard_log="./PPO2_tensorboard/")
# model.learn(total_timesteps=100000)
# model.save("with_indicator_100000_FIXED.pkl")
model.load("with_indicator_100000_FIXED.pkl")


# In[131]:

obs = env_test.reset()
balance_plot = list()
share_plot = list()
profit_plot = list()
maindata = list()

obs_plot = list()
n = 1000
for i in range(n):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env_test.step(action)
    obs_plot.append(obs)

    balance, profit, share = env_test.render()
    print(action)
    balance_plot.append(balance)
    share_plot.append(share)
    profit_plot.append(profit)
    maindata.append(df_test["Open"].iloc[i:i+1].values)
    if done:
        break


# In[132]:


plt.plot(np.array(profit_plot).ravel(),color='red')
plt.show()


# In[133]:


plt.plot(np.array(share_plot).ravel(), linewidth=1)
plt.show()


# In[134]:
#
# #
df_test[:n].plot(x='Date', y=['Open'], figsize=(10,5), grid=True)
plt.show()
fig = plt.figure()
plt.plot(np.array(maindata).ravel(), linewidth=1)
plt.plot(np.array(share_plot).ravel(), linewidth=1)
plt.grid(axis='both',linestyle='--')
plt.show()




