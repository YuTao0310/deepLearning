from ctgan.synthesizers.base import BaseSynthesizer
from ctgan import load_demo
from sdv import metadata

data = load_demo()

ctgan1 = BaseSynthesizer.load('./sdv/ctgan_epoch300.pt')
ctgan2 = BaseSynthesizer.load('./sdv/ctgan_aliyun.pt')

samples1 = ctgan1.sample(32563)
samples2 = ctgan2.sample(32563)

from sdv.evaluation import evaluate
from sdv import Metadata
m = Metadata()
m.add_table
evaluate(samples1, data, metrics=['BinaryDecisionTreeClassifier'], aggregate=False, metadata=metadata)