# GBFG
Qilin Li, Zhong Yuan, Dezhong Peng, Xiaomin Song, Huiming Zheng, and **Xinyu Su***, [Granular-ball fuzzy information-based outlier detector](https://doi.org/10.1016/j.ijar.2025.109473), International Journal of Approximate Reasoning, 2025.

## Abstract
Outlier detection is an important part of the process of carrying out data mining and analysis and has been applied to many fields. Existing methods are typically anchored in a single-sample processing paradigm, where the processing unit is each individual and single-granularity sample. This processing paradigm is inefficient and ignores the multi-granularity features inherent in data. In addition, these methods often overlook the uncertainty information present in the data. To remedy the above-mentioned shortcomings, we propose an unsupervised outlier detection method based on Granular-Ball Fuzzy Granules (GBFG). GBFG adopts a granular-ball-based computing paradigm, where the fundamental processing units are granular-balls. This shift from individual samples to granular-balls enables GBFG to capture the overall data structure from a multi-granularity perspective and improve the performance of outlier detection. Subsequently, we calculate the outlier factor based on the outlier degrees of the granular-ball fuzzy granules to which the sample belongs, serving as a measure of the outlier degrees of samples. The experimental results prove that GBFG has a remarkable performance compared with the existing excellent algorithms. The code of GBFG is publicly available on https://github.com/Mxeron/GBFG.

## Usage
You can run GBFG.py:
```python
if __name__ == '__main__':
    data = pd.read_csv("./Example.csv").values
    ID = (data >= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        data[:, ID] = scaler.fit_transform(data[:, ID])
    out_factors = GBFG(data, 0.4)
    print(out_factors)
```
You can get outputs as follows:
```
out_factors = [0.0737184  0.09715414 0.06793347 0.09525602 0.07668188 0.07749797
 0.08686666 0.08586372 0.08126347 0.08271278 0.08654951 0.07920195
 0.08677153 0.07670291 0.08176544 0.08937062 0.08070382 0.08017381
 0.09027395 0.07769059]
```
## Citation
If you find GBFG useful in your research, please consider citing:
```
@article{li2025granular,
  title={Granular-ball fuzzy information-based outlier detector},
  author={Li, Qilin and Yuan, Zhong and Peng, Dezhong and Song, Xiaomin and Zheng, Huiming and Su, Xinyu},
  journal={International Journal of Approximate Reasoning},
  pages={109473},
  year={2025},
  publisher={Elsevier}
}
```
## Contact
If you have any questions, please contact suxinyu@stu.scu.edu.cn or yuanzhong@scu.edu.cn.
