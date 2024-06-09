# RainbowCake Sentiment Analysis
- Lambda function repo: https://github.com/IntelliSys-Lab/RainbowCake-ASPLOS24/tree/master/applications/python_sentiment_analysis/src
- Main Library Repo: 
    - https://github.com/nltk/nltk
    - https://github.com/sloria/TextBlob

Optimization: Lazy load nltk.corpus, nltk.sem, nltk.stem, nltk.parse, nltk.tag <br>
#### Directories changed
- nltk
- textblob

### Average Initialization latency of
- Original code: 1453.747554347826
- Optimized code: 1073.6022012578617

#### Average Intialization latency reduced: 26.15%

### Average End to End latency of
- Original code: 1553.2020923913044
- Optimized code: 1171.361509433962

#### Average End to End latency reduced: 24.58%

### Average Memory Utilization of
- Original code: 132.6358695652174
- Optimized code: 123.0125786163522

#### Average Memory Utilization reduced: 7.25%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)