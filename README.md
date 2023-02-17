# 数据流常见数据集及其特征
> 为了更好的方便各位同学快速的进行数据流的实验，这里将常见的数据流数据及其处理方式.

> 首先，为了更好的规范数据流处理流程，这里介绍一个基于python的在线学习库`River`，`River`中包含了不少数据流算法
> 以及部分数据流和数据流生成器。接下来介绍一个使用river进行分类的示例.

> 所有数据下载连接：[Google](https://drive.google.com/drive/folders/1zqxdsfdPXZlZNFbLiPqtcuszgiHLc6pX?usp=sharing) , [阿里云盘](https://www.aliyundrive.com/s/u7Wo2HBbdUE)
> 
> ps:由于阿里云盘无法分享压缩包，所里这里采用`7-zip`的自释放程序方法进行压缩，直接运行分享的程序即可自动解压。
## 示例
```python
from river.datasets import synth
from river.preprocessing import StandardScaler
from river.ensemble import AdaBoostClassifier
from river.tree import HoeffdingTreeClassifier
from river.metrics import Accuracy
from sklearn.metrics import classification_report

dataset = synth.Hyperplane(n_features=10, n_drift_features=4)
scaler = StandardScaler()
model = AdaBoostClassifier(
    model=(
        HoeffdingTreeClassifier(
            split_criterion='gini',
            delta=1e-5,
            grace_period=2000
        )),
    n_models=5)

acc = Accuracy
y_true = []
y_pred = []
for x, y in dataset.take(1000):
    x = scaler.learn_one(x).transform_one(x)
    y_hat = model.predict_one(x)
    model.learn_one(x, y)
    
    acc.update(y, y_hat)
    y_true.append(y)
    y_pred.append(y_hat)
report = classification_report(y_true, y_pred)
```
## 真实数据集
### 1. Electricity
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/elecNormNew/elecNormNew.arff'
des_path = 'benchmark/realworld/elecNormNew/elecNormNew.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.iloc[:,-1] = df.iloc[:,-1].map(lambda x: 0 if x ==b'UP' else 1)
df.to_csv(des_path, index=False)
```
### 2. Forest Covertype
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/covtypeNorm/covtypeNorm.arff'
des_path = 'benchmark/realworld/covtypeNorm/covtypeNorm.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 3. Airlines
```python
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

encoder = OrdinalEncoder()
src_path = 'benchmark/realworld/airlines/airlines.arff'
des_path = 'benchmark/realworld/airlines/airlines.csv'
data = arff.loadarff(src_path)
data = pd.DataFrame(data[0]).values
encoder.fit(data[:,:4])
data = np.hstack([encoder.transform(data[:,:4]), data[:,4:]])
df = pd.DataFrame(data, dtype=float)
df.to_csv(des_path, index=False)
```
### 4. Poker

```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/poker-lsn/poker-lsn.arff'
des_path = 'benchmark/realworld/poker-lsn/poker-lsn.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 5. Shuttle

```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/poker-lsn/poker-lsn.arff'
des_path = 'benchmark/realworld/poker-lsn/poker-lsn.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 6. Twitter
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/twitter/twitter.arff'
des_path = 'benchmark/realworld/twitter/twitter.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 7. SPAM
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/spam/spam.arff'
des_path = 'benchmark/realworld/spam/spam.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 0 if x == b'spam' else 1)
df.to_csv(des_path, index=False)
```
### 8. Weather
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/weather/weather.arff'
des_path = 'benchmark/realworld/weather/weather.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 9. Gas
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/gas/gas.arff'
des_path = 'benchmark/realworld/gas/gas.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 10. Sensor
```python
from scipy.io import arff
import pandas as pd

src_path = 'benchmark/realworld/sensor/sensor.arff'
des_path = 'benchmark/realworld/sensor/sensor.csv'
data = arff.loadarff(src_path)
df = pd.DataFrame(data[0], dtype=float)
df.to_csv(des_path, index=False)
```
### 11. KDDCup
### 12. Outdoor
```python
import pandas as pd

data_path = 'benchmark/realworld/outdoor/outdoor.data'
label_path = 'benchmark/realworld/outdoor/outdoor.labels'
des_path = 'benchmark/realworld/outdoor/outdoor.csv'

data = pd.read_csv(data_path, sep=' ')
label = pd.read_csv(label_path)
df = pd.concat((data, label), axis=1)
header = []
for i in range(len(df.columns) - 1):
    header.append(f'f{i}')
header.append('classes')
df.to_csv(des_path, index=False, header=header)
```
### 13. Rialito
```python
import pandas as pd

data_path = 'benchmark/realworld/rialto/rialto.data'
label_path = 'benchmark/realworld/rialto/rialto.labels'
des_path = 'benchmark/realworld/rialto/rialto.csv'

data = pd.read_csv(data_path, sep=' ')
label = pd.read_csv(label_path)
df = pd.concat((data, label), axis=1)
header = []
for i in range(len(df.columns) - 1):
    header.append(f'f{i}')
header.append('classes')
df.to_csv(des_path, index=False, header=header)
```
### Statistic Information
```python
import os
import pandas as pd
from collections import Counter

columns = ['dataset','features','classes','instances','majority', 'minority']
info = pd.DataFrame(columns=columns)

dir = 'benchmark/realworld'
for name in os.listdir(dir):
    counter = Counter()
    path = f'{dir}/{name}/{name}.csv'

    data = pd.read_csv(path)
    data[data.columns[-1]].astype(int).apply(lambda x:counter.update([x]))
    stat = {
        'dataset': name,
        'features': data.shape[1],
        'classes': len(counter),
        'instances': data.shape[0],
        'majority': max(counter.values()) / data.shape[0] * 100,
        'minority': min(counter.values()) / data.shape[0] * 100
    }
    print(stat, sum(counter.values()))
```
|    Dataset     | Features | Classes | Instances | Majority | Minority |
|:--------------:|:--------:|:-------:|:---------:|:--------:|:--------:|
|    airlines    |    8     |    2    |  539383   |  55.46   |  44.54   |
|  covtypeNorm   |    55    |    7    |  581012   |  48.76   |   0.47   |
|  elecNormNew   |    9     |    2    |   45312   |  57.55   |  42.45   |
|      gas       |   129    |    6    |   13910   |  21.63   |  11.80   |
|     kddcup     |    42    |   23    |  494020   |  56.84   |   0.00   |
|    outdoor     |    22    |   40    |   3999    |   2.50   |   2.48   |
|   poker-lsn    |    11    |   10    |  829201   |  50.11   |   0.00   |
|     rialto     |    28    |   10    |   82249   |  10.00   |   9.99   |
|     sensor     |    6     |   55    |  2219803  |   2.96   |   0.09   |
|    shuttle     |    10    |    7    |   57999   |  78.60   |   0.02   |
|      spam      |   500    |    2    |   9324    |  74.40   |  25.60   |
|    twitter     |    31    |    2    |   9090    |  84.29   |  15.71   |
|    weather     |    9     |    2    |   18159   |  68.62   |  31.38   |
### References and Reproducible
[MOA:https://moa.cms.waikato.ac.nz/datasets](https://moa.cms.waikato.ac.nz/datasets/)

[github:https://github.com/vlosing/driftDatasets](https://github.com/vlosing/driftDatasets)

[Benchmark experiments result(very import)](https://people.vcu.edu/~acano/imbalanced-streams/)

## Aritficial Datasets
```python
from river.datasets import synth

dataset = synth.ConceptDriftStream(
    stream=synth.SEA(seed=42, variant=0),
    drift_stream=synth.SEA(seed=42, variant=1),
    seed=1, position=5, width=2
)

for x, y in dataset.take(10):
    print(x, y)
```
```python
# River中数据集生成器种类
__all__ = [
"Agrawal",
"AnomalySine",
"ConceptDriftStream",
"Friedman",
"FriedmanDrift",
"Hyperplane",
"LED",
"LEDDrift",
"Logical",
"Mixed",
"Mv",
"Planes2D",
"RandomRBF",
"RandomRBFDrift",
"RandomTree",
"SEA",
"Sine",
"STAGGER",
"Waveform",
]
```
