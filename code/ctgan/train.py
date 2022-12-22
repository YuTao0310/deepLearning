from ctgan import CTGANSynthesizer
import pandas as pd
import sys

sys.stdout = open('recode.log', mode = 'w',encoding='utf-8')

# Names of the columns that are discrete
discrete_columns = [
    'Morphological impact value (G)',
    'The richness of sand source (F)'
]

data1 = pd.read_excel('Data_generation.xlsx', sheet_name='east')
ctgan = CTGANSynthesizer(epochs=500, batch_size=50, pac=5, verbose=True)
ctgan.fit(data1, discrete_columns)
ctgan.save("ctgan_chang1_windows.pt")

# Synthetic copy
samples1 = ctgan.sample(1000)

data2 = pd.read_excel('Data_generation.xlsx', sheet_name='west')
ctgan = CTGANSynthesizer(epochs=500, batch_size=50, pac=5, verbose=True)
ctgan.fit(data2, discrete_columns)
ctgan.save("ctgan_chang2_windows.pt")

# Synthetic copy
samples2 = ctgan.sample(1000)

writer = pd.ExcelWriter('synthetic_data.xlsx')
samples1.round(2).to_excel(writer, index=None, sheet_name='east')
samples2.round(2).to_excel(writer, index=None, sheet_name='west')
writer.save()
writer.close()