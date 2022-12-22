from sdv.demo import load_tabular_demo

from sdv.tabular import GaussianCopula

real_data = load_tabular_demo('student_placements')

model = GaussianCopula()

model.fit(real_data)

synthetic_data = model.sample()