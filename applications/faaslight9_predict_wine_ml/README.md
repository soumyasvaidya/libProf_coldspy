- Application Name: FaaSLight 9

Optimization: Removed the use of Pandas
Did not remove Numpy because it is needed to accelerate model training and serving.
It is also needed for sklearn.

```
faaslight9_predict_wine_ml
986
897
Average of exec_before is : 392.2805172413793
Average of exec_after is : 353.84502787068004
Percent difference is : -9.797960306820187
Average of init_before is : 4762.057910750507
Average of init_after is : 2705.3339464882943
Percent difference is : -43.18981421076566
Average of e2e_before is : 5154.3384279918855
Average of e2e_after is : 3059.1789743589743
Percent difference is : -40.64846503393412
Average of Max Memory used before is : 251.9077079107505
Average of Max Memory used after is : 187.76254180602007
Percent difference is : -25.463756800747323
```


## 100+ Cold Starts
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)
