from ctgan.synthesizers.base import BaseSynthesizer
from ctgan import load_demo
from sdv.metrics.tabular import CSTest, KSTest

data = load_demo()

ctgan1 = BaseSynthesizer.load('./sdv/ctgan_epoch300.pt')
ctgan2 = BaseSynthesizer.load('./sdv/ctgan_aliyun.pt')

samples1 = ctgan1.sample(32563)
samples2 = ctgan2.sample(32563)

print(f"CSTest: ctgan1-{CSTest.compute(data, samples1)}")
print(f"CStest: ctgan2-{CSTest.compute(data, samples2)}")
print(f"KSTest: ctgan1-{KSTest.compute(data, samples1)}")
print(f"KStest: ctgan2-{KSTest.compute(data, samples2)}")