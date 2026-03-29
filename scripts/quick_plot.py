import matplotlib.pyplot as plt
import pandas as pd

exp_file = "data/processed/gaussian_matched/AMPPUL30.parquet"
data_exp = pd.read_parquet(exp_file)
t_exp = data_exp.iloc[:, 0].to_numpy()
v0_exp = data_exp.iloc[:, 1].to_numpy()
v40_exp = data_exp.iloc[:, 2].to_numpy()


plt.figure()
plt.plot(t_exp, v0_exp, label="Experimental $V_{0}$")
plt.plot(t_exp, v40_exp, label="Experimental $V_{40}$")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

plt.show()
