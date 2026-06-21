## Experiments

### Experimental Settings

​    云-边-端拓扑
​    节点 CPU、计算速率、可靠性、功率
​    链路带宽、链路可靠性、传输能耗
​    动态可靠性参数
​    种群规模、迭代次数、变异概率 【增加超参数选择说明】
​    每组实验重复多次取平均值

### Compared Algorithms

Random
MaxResource / LB
LocalFirst / LPD
RTBL
RDODA-DSSA
C-RDODA-DSSA    

### Pareto Front Quality

Hypervolume
Pareto 前沿分布

### Overall Performance under Different Number of DNNs 

平均动态运行可靠性
平均动态精度可靠性
平均估计推理时延
平均总能耗 增加说明
拒绝/失败部署数
运行时间
    

### Dynamic Reliability Evaluation

设置弱、中、强三种动态退化场景
改变：
    节点可靠性退化参数
    链路可靠性退化参数
    热度遗忘系数
观察：
    动态运行可靠性
    动态精度可靠性
    失败部署数
目的：
	证明算法能避开高负载、高热度、低可靠节点。 

### Ablation Study 

Full C-RDODA / RDODA-->NSGA-II
w/o DAG-aware initialization
w/o DAG block cross-over
w/o skew mutation 
w/o elite local search 

### Parameter Sensitivity

遗传算法超参数

### Large-scale Network Evaluation

验证算法在节点数量、链路数量、DNN 请求数量和任务规模同时增大时的有效性与可扩展性。
    Small: 1 cloud + 10 edge + 30 users 
    Medium: 1 cloud + 30 edge + 120 users 
    Large: 1 cloud + 60 edge + 300 users 
    同时需要等比放大 DNN 数量：
    
不同网络规模下的算法性能
	可靠性是否随规模扩大保持稳定
	运行时间是否近似可接受增长
	失败部署数是否下降或保持低水平
	算法是否能利用更多边缘资源降低时延和能耗
不同请求负载下的算法性能
	动态可靠性是否明显下降
	拒绝数量增长是否低于基线
	能耗和时延是否仍可控
	C-RDODA-DSSA 是否比 RDODA-DSSA 更稳定
大规模网络下的运行时间扩展性

### 5.11 Application Preference Scenarios

构造四类任务偏好：

- Delay-sensitive 
- Reliability-sensitive
- Energy-sensitive
- Balanced

证明算法不仅能产生 Pareto 解集，还能根据不同服务需求选择合适部署方案。