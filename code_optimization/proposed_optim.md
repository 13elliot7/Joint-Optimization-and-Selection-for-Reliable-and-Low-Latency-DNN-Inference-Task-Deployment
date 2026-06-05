![image-20260517151329100](/Users/hlz/Library/Application Support/typora-user-images/image-20260517151329100.png)



![image-20260517151343449](/Users/hlz/Library/Application Support/typora-user-images/截屏2026-05-17 15.14.28.png)



Get_res

```python
    def get_res(self, i: int, pq: List[List[List[int]]], values1: List[float], values2: List[float], values3: List[float]) -> List[List[int]]:
        """按拥挤距离规则计算同层个体的距离值。"""
        v1 = list(range(len(values1)))
        v2 = list(range(len(values2)))
        v3 = list(range(len(values3))) # 下标和序号对应

        # 冒泡排序
        for m in range(len(v1)):
            for n in range(len(v1) - 1 - m):
                if values1[n] > values1[n + 1]:
                    values1[n], values1[n + 1] = values1[n + 1], values1[n]
                    v1[n], v1[n + 1] = v1[n + 1], v1[n]
        for m in range(len(v2)):
            for n in range(len(v2) - 1 - m):
                if values2[n] > values2[n + 1]:
                    values2[n], values2[n + 1] = values2[n + 1], values2[n]
                    v2[n], v2[n + 1] = v2[n + 1], v2[n]
        for m in range(len(v3)):
            for n in range(len(v3) - 1 - m):
                if values3[n] > values3[n + 1]:
                    values3[n], values3[n + 1] = values3[n + 1], values3[n]
                    v3[n], v3[n + 1] = v3[n + 1], v3[n]


        for m in range(len(pq)):
            if v1[m] == 0 or v1[m] == len(values1) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values1[v1[m] + 1] - values1[v1[m] - 1]), values1[-1] - values1[0])
        for m in range(len(pq)):
            if v2[m] == 0 or v2[m] == len(values2) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values2[v2[m] + 1] - values2[v2[m] - 1]), values2[-1] - values2[0])
        for m in range(len(pq)):
            if v3[m] == 0 or v3[m] == len(values3) - 1:
                self.distance[i][m] = 2**31 - 1
            else:
                self.distance[i][m] += _java_div(abs(values3[v3[m] + 1] - values3[v3[m] - 1]), values3[-1] - values3[0])
        return self.distance
```



 如果改成四目标：

  - 候选解只要在能耗上明显更优，就更不容易被别的解支配
  - 节能解会更早进入高 Pareto 层，而不是只在最后一轮“加分”

  结果是：

  - 搜索会更主动保留低能耗方案
  - 不再只是从“可靠性/时延已经不错的解里”挑一个更省能的
  - 而是整个种群会朝“可靠性、时延、能耗都平衡”的方向演化