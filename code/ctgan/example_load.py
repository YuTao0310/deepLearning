from ctgan.synthesizers.base import BaseSynthesizer

'''
将输出信息存到日志文件中
'''
import sys
sys.stdout = open('./ctgan/ctgan_epoch300_record.log', mode = 'w',encoding='utf-8')

ctgan = BaseSynthesizer.load('./ctgan/ctgan_epoch300.pt')

samples = ctgan.sample(1000)

''' 
行不受限制显示
'''
import pandas as pd
pd.set_option('display.max_rows', None)

print(samples)