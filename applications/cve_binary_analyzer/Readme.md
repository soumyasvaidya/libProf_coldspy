# CVE-BIN-TOOL
- Lambda function repo: https://github.com/intel/cve-bin-tool/tree/main
- Main Library Repo: https://github.com/sissaschool/xmlschema

#### How to run:
After installing dependencies change the paths of the following files from the cve_bin_tool library
- cve_bin_tool/cvedb.py
- cve_bin_tool/datasources/__init__.py
At the first run, do not use offline mode

#### Optimization: Lazy load xmlschema <br>
Files changed
- cve-bin-tool/validator.py:11

### Average Initialization latency of
- Original code: 2489.4658269230767
- Optimized code: 1965.2653461538464

#### Average Intialization latency reduced: 21.06%

### Average End to End latency of
- Original code: 3328.6189615384615
- Optimized code: 2785.409711538461

#### Average End to End latency reduced: 16.32%

### Average Memory Utilization of
- Original code: 109.525
- Optimized code: 99.55384615384615

#### Average Memory Utilization reduced: 9.10%

## 500 Cold Starts CDF
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)
