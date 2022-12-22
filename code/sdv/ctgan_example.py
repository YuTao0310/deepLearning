from ctgan import load_demo
data = load_demo()

from sdv.tabular import CTGAN
model = CTGAN()
model.fit(data)