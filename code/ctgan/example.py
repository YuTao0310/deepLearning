from ctgan import CTGANSynthesizer
from ctgan import load_demo

import sys

sys.stdout = open('recode.log', mode = 'w',encoding='utf-8')

data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGANSynthesizer(epochs=1, verbose=True)
ctgan.fit(data, discrete_columns)
ctgan.save("ctgan_windows.pt")

# Synthetic copy
samples = ctgan.sample(1000)

h = 3
print(f"{h} is good")