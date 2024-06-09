# RainbowCake DNA Visualization
- Lambda function repo: https://github.com/IntelliSys-Lab/RainbowCake-ASPLOS24/tree/master/applications/python_dna_visualization/src
- Main Library Repo: https://github.com/IQTLabs/squiggle

Optimization: Removed Numpy

### Average Initialization latency of
- Original code: 842.4089980353634
- Optimized code: 365.61019379844964

#### Average Intialization latency reduced: 56.59%

### Average End to End latency of
- Original code: 861.7379371316306
- Optimized code: 381.41406976744184

#### Average End to End latency reduced: 55.73%

### Average Memory Utilization of
- Original code: 76.18467583497053
- Optimized code: 53.0

#### Average Memory Utilization reduced: 30.43%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)