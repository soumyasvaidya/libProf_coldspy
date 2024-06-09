# RainbowCake Graph MST
- Lambda function repo: https://github.com/IntelliSys-Lab/RainbowCake-ASPLOS24/tree/master/applications/python_graph_mst/src
- Main Library Repo: https://github.com/igraph/python-igraph

Optimization: Lazy load igraph.drawing, igraph.io, igraph.clustering <br>
#### Files changed
- igraph/__init__.py
- igraph/clustering.py
- igraph/community.py
- igraph/layout.py
- igraph/io/images.py

### Average Initialization latency of
- Original code: 293.59251509054326
- Optimized code: 168.9913141025641

#### Average Intialization latency reduced: 42.44%

### Average End to End latency of
- Original code: 301.59144869215294
- Optimized code: 177.22878205128205

#### Average End to End latency reduced: 41.24%

### Average Memory Utilization of
- Original code: 42.96378269617706
- Optimized code: 39.07051282051282

#### Average Memory Utilization reduced: 9.06%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)