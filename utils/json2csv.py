import pandas as pd
df = pd.read_json('submission.json')
print(df.shape)
df.to_csv("submission.csv",index=None)