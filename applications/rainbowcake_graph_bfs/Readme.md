# RainbowCake Graph BFS
- Lambda function repo: https://github.com/IntelliSys-Lab/RainbowCake-ASPLOS24/tree/master/applications/python_graph_bfs/src
- Main Library Repo: https://github.com/igraph/python-igraph

Optimization: Lazy load igraph.drawing, igraph.io, igraph.clustering <br>
#### Files changed
- igraph/__init__.py
- igraph/clustering.py
- igraph/community.py
- igraph/layout.py
- igraph/io/images.py

### Average Initialization latency of
- Original code: 292.5434226804124
- Optimized code: 171.35102880658437

#### Average Intialization latency reduced: 41.43%

### Average End to End latency of
- Original code: 300.4947422680412
- Optimized code: 179.4524074074074

#### Average End to End latency reduced: 40.28%

### Average Memory Utilization of
- Original code: 42.97731958762886
- Optimized code: 39.150205761316876

#### Average Memory Utilization reduced: 8.90%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)