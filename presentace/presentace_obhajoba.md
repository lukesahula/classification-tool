<!-- $theme: gaia -->
<!-- $size: 16:9 -->

<p align="center">
<img src="./a_random_forest.png"/> <br />
</p>

---

<p align="center">
<img src="./a_random_forest.png"/> <br />
Random forest, <b>source:</b> http://blog.yhat.com
</p>

---

<br />
<h3 align="center">Handling Missing Values in Decision Forests</h3>
<h3 align="center">in the Encrypted Network Traffic</h3>
<br />
<br />

<div align="right">
	Author: Lukáš Sahula <br />
	Supervisor: Ing. Jan Brabec <br />
	Bachelor thesis <br /> <br />
    Czech Technical University in Prague <br />
    Faculty of Electrical Engineering <br />
    Department of Computer Science <br />
</div>


---

<!-- page_number: true -->
### Handling missing values...
<br />
<br />
<table align="center">
<thead>
<tr>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Name</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Animal</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Age</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Gender</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Description</font></td>
</tr>
</thead>
<tbody>
<tr>
<td align="right">Rex</td>
<td align="right">Dog</td>
<td align="right">3</td>
<td align="right">Male</td>
<td align="right">A good boy</td>
</tr>
<tr>
<td align="right">Lady</td>
<td align="right">Dog</td>
<td align="right" style="color: red;">X</td>
<td align="right">Female</td>
<td align="right" style="color: red;">X</td>
</tr>
<tr>
<td align="right">Cat</td>
<td align="right">Cat</td>
<td align="right">4</td>
<td align="right">Male</td>
<td align="right" style="color: red;">X</td>
</tr>
<tr>
<td align="right">Kitty</td>
<td align="right">Cat</td>
<td align="right" style="color: red;">X</td>
<td align="right">Female</td>
<td align="right">Likes to cuddle</td>
</tr>
<tr>
<td align="right">Gizmo</td>
<td align="right" style="color: red">X</td>
<td align="right" style="color: red;">X</td>
<td align="right">Male</td>
<td align="right" style="color: red;">X</td>
</tr>
</tbody>
</table>

---

### ... in Decision Forests ...
<p align="center">
	<img src="./dtree.png" style="width: 80%"/> <br />
    Decision Tree classifier, <b>source:</b> http://packtpub.com
</p>

---

### ... in the Encrypted Network Dataset
- Data from network proxy logs
- Classification of malware 
- Over 100 classes of malware
- 50 features
- Data missingness over 50%

---

### Dataset correlation analysis
<p align="center">
	<img src="./corr.png" style="width: 44%"/> <br />
    Heatmap of feature pairs correlations (Pearson)
</p>

---

### Conditional probabilities of missingness
<p align="center">
	<img src="./cond.png" style="width: 44%"/> <br />
    P(i_missing | j_not_missing)
</p>

---

### Feature substitution
<p align="center">
	<img src="./corr_thres.png" style="width: 44%"/> <br />
	Feature pairs with PCC above 0.3
</p>

---

### Existing methods for missing data imputation
- <b>Baseline</b>
- <b>Strawman imputation (mean or median)</b>
- <b>On-the-fly-imputation method</b>
- <b>Missingness incorporated in attributes</b>
- MissForest
- Surrogate splits
- ...

---

### Evaluation metrics
<p align="center">
	<img src="./precrec.png" /> <br />
    Precision and recall, <b>source:</b> http://wikipedia.org
</p>

---

### Experiments with random forests
- Number of trees: 100
- Minimal number of samples for a split: 2
- Maximal number of features for a split: sqrt
- Maximal depth of trees: unlimited
- Trained on data from three days in January 2017
- Tested on data from one day in March 2017
- Randomness factor: 1% of variance in recall and precision

---

### Results
<br />
<table align="center">
<thead>
<tr>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Method</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Precision</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Recall</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Prec = 1.0</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Prec > 0.8</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Prec > 0.5</font></td>

</tr>
</thead>
<tbody>
<tr>
<td align="right">Baseline</td>
<td align="right">0.61</td>
<td align="right">0.57</td>
<td align="right">22</td>
<td align="right">54</td>
<td align="right">70</td>
</tr>
<tr>
<td align="right">Mean</td>
<td align="right">0.59</td>
<td align="right">0.54</td>
<td align="right">21</td>
<td align="right">54</td>
<td align="right">70</td>
</tr>
<tr>
<td align="right">Median</td>
<td align="right">0.56</td>
<td align="right">0.49</td>
<td align="right">19</td>
<td align="right">45</td>
<td align="right">65</td>
</tr>
<tr>
<td align="right">OTFI</td>
<td align="right" style="color: red">0.23</td>
<td align="right" style="color: red">0.06</td>
<td align="right" style="color: red">18</td>
<td align="right" style="color: red">25</td>
<td align="right" style="color: red">25</td>
</tr>
<tr>
<td align="right">MIA</td>
<td align="right" style="color: blue">0.65</td>
<td align="right" style="color: blue">0.58</td>
<td align="right" style="color: blue">28</td>
<td align="right" style="color: blue">60</td>
<td align="right" style="color: blue">74</td>
</tr>
</tbody>
</table>
<p align="center">Average precision, recall, and number of classes with precision above a certain threshold</p>

---

### Contributions
- Correlation of datasets features analysed
- Algorithms compared on real data
- On-the-fly-imputation found not suited for data with heavy missingness
- Missingness incorporated in attributes slightly improves the baseline method
- Python framework for further experiments implemented

---

### Answers
#### Method speed comparison
- Baseline: ~18 hours
- Strawman: ~18 hours
- MIA: ~45 hours
- OTFI: ~100 hours

