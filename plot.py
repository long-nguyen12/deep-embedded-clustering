import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

idec = pd.read_csv('./results/idec/idec_log.csv')
dec = pd.read_csv('./results/dec_log.csv')

ep = []
for i in range(1,40):
  ep.append(i)

val_idec_acc = idec[['acc']]

val_dec_acc = dec[['acc']]

plt.plot(val_idec_acc[0:40], label="IDEC", marker='d',markevery=10, color='b', ms = 5)
plt.plot(val_dec_acc[0:40], label="DEC", linestyle="dotted", marker='o', ms = 5,markevery=10)
plt.title('Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
