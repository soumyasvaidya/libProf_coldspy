# RainbowCake Graph Pagerank
- Lambda function repo: https://github.com/IntelliSys-Lab/RainbowCake-ASPLOS24/tree/master/applications/python_graph_pagerank/src
- Main Library Repo: https://github.com/igraph/python-igraph

Optimization: Lazy load igraph.drawing, igraph.io, igraph.clustering <br>
#### Files changed
- igraph/__init__.py
- igraph/clustering.py
- igraph/community.py
- igraph/layout.py
- igraph/io/images.py

### Average Initialization latency of
- Original code: 287.4685616438356
- Optimized code: 169.57597137014315

#### Average Intialization latency reduced: 41.01%

### Average End to End latency of
- Original code: 305.51821917808223
- Optimized code: 188.43519427402862

#### Average End to End latency reduced: 38.32%

### Average Memory Utilization of
- Original code: 43.17123287671233
- Optimized code: 39.23108384458078

#### Average Memory Utilization reduced: 9.13%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)