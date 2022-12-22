import pandas as pd
 
url = 'https://ranking.promisingedu.com/qs'
 
df_list = pd.read_html(url)

a = df_list[0
]
print(df_list[0])