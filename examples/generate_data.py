import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

import xlsxwriter

# import data
filepath = 'stock_prices.csv'
df = pd.read_csv(filepath, index_col='Index')
print(df.shape)  # (1771, 8)
df.tail()
# first difference
df_differenced = df.diff().dropna()

# fit model
model = VAR(df_differenced)

model_fitted = model.fit(12)

model_fitted.is_stable()
model_fitted.summary()

# simulate data
sim_fd = model_fitted.simulate_var(steps = 48*30+11)
df_sim_fd = pd.DataFrame(sim_fd[11:,:])
df_sim_fd.columns = df.columns
plt.plot(df_sim_fd)

test = VAR(sim_fd)
test_fit = test.fit(8)
test_fit.summary()

#add initial value
#df_sim = pd.concat([df.head(1).reset_index(drop=True), df_sim_fd], axis=0)

#df_sim = df_sim.cumsum()
df_sim = df_sim_fd
df_sim.columns = ["q_1", "q_2", "m_1", "m_2", "m_3","w_1", "w_2", "w_3"]

#aggregate to different levels data for mufbvar
df_sim_q = df_sim.iloc[:,[0,1]]
df_sim_q = df_sim_q.groupby(df_sim_q.index // 12).mean()
df_sim_q.columns = ["q_1", "q_2"]

df_sim_m = df_sim.iloc[:,[2, 3, 4]]
df_sim_m = df_sim_m.groupby(df_sim_m.index // 4).mean()
df_sim_m.columns = ["m_1", "m_2", "m_3"]


df_sim_w = df_sim.iloc[:,[5,6,7]]
df_sim_w.columns = ["w_1", "w_2", "w_3"]




# write data into excel

name = "hist.xlsx"
with pd.ExcelWriter(name) as writer:
#writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")

    df_sim_q.to_excel(writer, sheet_name="Q")
    df_sim_m.to_excel(writer, sheet_name="M")
    df_sim_w.to_excel(writer, sheet_name="W")
        


