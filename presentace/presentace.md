<!-- $theme: gaia -->
<!-- $size: 16:9 -->

#### Toolkit for malware classification
##### Lukáš Sahula


---
<!-- page_number: true -->
#### Something to break the ice
<p align="center">
<img src="./sweng.jpg" />
</p>

---

#### Classification pipeline
<p align="center">
	<img src="./pipeline.jpg" />
</p>

---

#### Toolkit classes
<p align="center">
	<img src="./classes.jpg" />
</p>

---

#### Evaluation metrics
<p align="center">
	<img src="./precrec.png" />
</p>

---

#### Baseline model

<br />
<table align="center">
<thead>
<tr>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Avg precision</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>Avg recall</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>TPs</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>FPs</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>FNs</font></td>
</tr>
</thead>
<tbody>
<tr>
<td align="right">0.28</td>
<td align="right">0.31</td>
<td align="right">174</td>
<td align="right">561</td>
<td align="right">533</td>
</tr>
</tbody>
</table>
<br>
<table align="center">
<thead>
<tr>
<td bgcolor="#2F2F2F"><font color=#fff8e1># of classes</font></td>
<td bgcolor="#2F2F2F"><font color=#fff8e1>with precision >=</font></td>
</tr>
</thead>
<tbody>
<tr>
<td align="right">10</td>
<td align="right">1.00</td>
</tr>
<tr>
<td align="right">10</td>
<td align="right">0.95</td>
</tr>
<tr>
<td align="right">10</td>
<td align="right">0.90</td>
</tr>
<tr>
<td align="right">12</td>
<td align="right">0.80</td>
</tr>
<tr>
<td align="right">19</td>
<td align="right">0.50</td>
</tr>
</tbody>
</table>

---

#### Future work
1. Implementing random forest
2. Dealing with missing values
3. Beating the baseline model