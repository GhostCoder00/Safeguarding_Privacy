# Digital Self-Control

> **[UMAP 2024] Digital Self-Control: A Solution for Enhancing Self-Regulation in Online Learning**.
> Digital Self-Control a video-based cognitive distraction detection system based on facial expressions and gaze features, which could aid learners in keeping their attention. It addresses the challenges associated with eyeglass wearers and improves the model performance with additional features containing information about the presence of glasses. It utilizes federated learning to safeguards users' data privacy. 

This repository is anonymized for review.

## 🖼️ Teaser
<img src="https://github.com/wmd0701/Digital-Self-Control/assets/34072813/6baafcb5-138e-4006-8b88-984842a052b5" width="700">

## 💁 Usage
1. Download data and carry out data preprocessing following the instructions below.

2. Create conda environment with `conda env create -f environment.yml`.

3. Run `experiments_nFL.py` for centralized learning and `experiments_FL.py` for federated learning.

For detailed argument settings please check `utils.py`. 

## 🔧 Environment
The experiments were conducted on multiple computers that are identical both on hardware and software levels (by **January 25th, 2024**:):
| Environment | Specs |
| --- | ----------- |
| CPU | Intel i7-13700K |
| GPU | NVIDIA RTX 4080 |
| RAM | 32 GB |
| OS | WSL2 Ubuntu 22.04 LTS |
| Python | 3.11.7 by Anaconda |
| PyTorch | 2.1.2 for CUDA 12.1 |
| TorchMetrics | 1.2.1 |
| Scikit-Learn | 1.4.0 |
| WandB | 0.16.2 |

Others:
- We used **[Weights & Bias](https://wandb.ai/site)** for figures instead of tensorboard. Please install and set up it properly beforehand.

- We used the Python function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## 🗺 Instructions on data preprocessing
We conducted experiments using four datasets: [Colorado](https://ieeexplore.ieee.org/abstract/document/8680698), [Korea](https://nmsl.kaist.ac.kr/projects/attention/), [Engagenet](https://github.com/engagenet/engagenet_baselines), and [DAISEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html). Please contact the corresponding authors for data access if necessary.

Please dive into the `data_preprocessing_and_feature_extraction` directory for further instructions on data preprocessing.

## 📈 Statistics
A brief summary about the exact data distribution concerning gender and the presence of glasses:
<table class="tg">
<thead>
  <tr>
    <th class="tg-xwyw" rowspan="3">Dataset<br></th>
    <th class="tg-c3ow" colspan="4"># samples of male participants</th>
    <th class="tg-c3ow" colspan="4"># samples of female participants</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="2">glass wearers</th>
    <th class="tg-c3ow" colspan="2">non-glass wearers</th>
    <th class="tg-c3ow" colspan="2">glass wearers</th>
    <th class="tg-c3ow" colspan="2">non-glass wearers</th>
  </tr>
  <tr>
    <th class="tg-c3ow">positive</th>
    <th class="tg-c3ow">negative</th>
    <th class="tg-c3ow">positive</th>
    <th class="tg-c3ow">negative</th>
    <th class="tg-c3ow">positive</th>
    <th class="tg-c3ow">negative</th>
    <th class="tg-c3ow">positive</th>
    <th class="tg-c3ow">negative</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">EngageNet</td>
    <td class="tg-c3ow">468</td>
    <td class="tg-c3ow">1506</td>
    <td class="tg-c3ow">1235</td>
    <td class="tg-c3ow">2894</td>
    <td class="tg-c3ow">478</td>
    <td class="tg-c3ow">562</td>
    <td class="tg-c3ow">622</td>
    <td class="tg-c3ow">1239</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Colorado</td>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">89</td>
    <td class="tg-c3ow">332</td>
    <td class="tg-c3ow">879</td>
    <td class="tg-c3ow">30</td>
    <td class="tg-c3ow">212</td>
    <td class="tg-c3ow">628</td>
    <td class="tg-c3ow">1128</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Korea</td>
    <td class="tg-c3ow">74</td>
    <td class="tg-c3ow">278</td>
    <td class="tg-c3ow">26</td>
    <td class="tg-c3ow">236</td>
    <td class="tg-c3ow">26</td>
    <td class="tg-c3ow">201</td>
    <td class="tg-c3ow">80</td>
    <td class="tg-c3ow">299</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DAiSEE</td>
    <td class="tg-c3ow">713</td>
    <td class="tg-c3ow">1606</td>
    <td class="tg-c3ow">1113</td>
    <td class="tg-c3ow">2796</td>
    <td class="tg-c3ow">228</td>
    <td class="tg-c3ow">1352</td>
    <td class="tg-c3ow">199</td>
    <td class="tg-c3ow">918</td>
  </tr>
</tbody>
</table>

Details about the exact number of positive samples, number of participants with glasses (gl) and non-glass wearers (ngl), and the gender of the participants in the train and test set. M and F are referring to the number of male and female participants.
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh"></th>
    <th class="tg-baqh"></th>
    <th class="tg-baqh" colspan="6">Train &amp; validation sets</th>
    <th class="tg-baqh" colspan="6">Test set</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">Dataset</td>
    <td class="tg-baqh">all</td>
    <td class="tg-baqh">pos</td>
    <td class="tg-baqh">gl</td>
    <td class="tg-baqh">ngl</td>
    <td class="tg-baqh">M</td>
    <td class="tg-baqh">F</td>
    <td class="tg-baqh">all</td>
    <td class="tg-baqh">pos</td>
    <td class="tg-baqh">gl</td>
    <td class="tg-baqh">ngl</td>
    <td class="tg-baqh">M</td>
    <td class="tg-baqh">F</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">non-FL</td>
    <td class="tg-baqh">EngageNet</td>
    <td class="tg-baqh">8193</td>
    <td class="tg-baqh">2495</td>
    <td class="tg-baqh">29</td>
    <td class="tg-baqh">54</td>
    <td class="tg-baqh">58</td>
    <td class="tg-baqh">25</td>
    <td class="tg-baqh">811</td>
    <td class="tg-baqh">308</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">12</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">9</td>
  </tr>
  <tr>
    <td class="tg-baqh">Colorado</td>
    <td class="tg-baqh">3001</td>
    <td class="tg-baqh">895</td>
    <td class="tg-baqh">9</td>
    <td class="tg-baqh">108</td>
    <td class="tg-baqh">46</td>
    <td class="tg-baqh">71</td>
    <td class="tg-baqh">302</td>
    <td class="tg-baqh">100</td>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">10</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">9</td>
  </tr>
  <tr>
    <td class="tg-baqh">Korea</td>
    <td class="tg-baqh">979</td>
    <td class="tg-baqh">170</td>
    <td class="tg-baqh">6</td>
    <td class="tg-baqh">6</td>
    <td class="tg-baqh">5</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">241</td>
    <td class="tg-baqh">36</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">1</td>
  </tr>
  <tr>
    <td class="tg-baqh">DAiSEE</td>
    <td class="tg-baqh">7835</td>
    <td class="tg-baqh">2095</td>
    <td class="tg-baqh">46</td>
    <td class="tg-baqh">54</td>
    <td class="tg-baqh">73</td>
    <td class="tg-baqh">27</td>
    <td class="tg-baqh">1090</td>
    <td class="tg-baqh">158</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">4</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">FL</td>
    <td class="tg-baqh">EngageNet</td>
    <td class="tg-baqh">8193</td>
    <td class="tg-baqh">2495</td>
    <td class="tg-baqh">29</td>
    <td class="tg-baqh">54</td>
    <td class="tg-baqh">58</td>
    <td class="tg-baqh">25</td>
    <td class="tg-baqh">811</td>
    <td class="tg-baqh">308</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">12</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">9</td>
  </tr>
  <tr>
    <td class="tg-baqh">Colorado</td>
    <td class="tg-baqh">2998</td>
    <td class="tg-baqh">905</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">109</td>
    <td class="tg-baqh">45</td>
    <td class="tg-baqh">72</td>
    <td class="tg-baqh">305</td>
    <td class="tg-baqh">90</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">9</td>
    <td class="tg-baqh">5</td>
    <td class="tg-baqh">8</td>
  </tr>
  <tr>
    <td class="tg-baqh">Korea</td>
    <td class="tg-baqh">979</td>
    <td class="tg-baqh">191</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">5</td>
    <td class="tg-baqh">6</td>
    <td class="tg-baqh">6</td>
    <td class="tg-baqh">241</td>
    <td class="tg-baqh">15</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">2</td>
  </tr>
  <tr>
    <td class="tg-baqh">DAiSEE</td>
    <td class="tg-baqh">7835</td>
    <td class="tg-baqh">2095</td>
    <td class="tg-baqh">46</td>
    <td class="tg-baqh">54</td>
    <td class="tg-baqh">73</td>
    <td class="tg-baqh">27</td>
    <td class="tg-baqh">1090</td>
    <td class="tg-baqh">158</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">4</td>
  </tr>
</tbody>
</table>

## 🏃 Hyperparameter search
The training data was divided into 5 folds for cross-validation. We conducted a grid search in range $10^{-5.5}, 10^{-5}, 10^{-4.5}, 10^{-4}, 10^{-3.5}, 10^{-3}, 10^{-2.5}, 10^{-2}, 10^{-1.5}, 10^{-1}$ for optimal learning rates with the help of cross-validation. When searching for learning rates for FL methods, we first looked for the optimal client learning rates for FedAvg and then applied this learning rate for all other methods. For FL algorithms that require a server optimizer, such as FedAdam, FedAwS, and TurboSVM-FL, we carried out a grid search for server learning rates in the same range. The mini-batch size was 4 throughout the experiments, and any user with less than 4 samples was left out for the experiments. The number of client local training epochs was set to 8. For centralized learning, the optimizer was set to stochastic gradient descent (SGD). For FL, we chose SGD as the client optimizer and Adam as the server optimizer, as suggested in previous works. A further common assumption in FL is that not all clients can participate in every global aggregation round. In this regard, we assumed 50\% of clients attended each aggregation round, and we randomly sampled these clients in each round. The best-performing learning rates are given in the table below.
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix">Features</th>
    <th class="tg-nrix">Algorithm</th>
    <th class="tg-nrix">Colorado</th>
    <th class="tg-nrix">Korea</th>
    <th class="tg-nrix">EngageNet</th>
    <th class="tg-nrix">DAiSEE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="8">EmoNet + <br>OpenFace<br>gaze</td>
    <td class="tg-nrix">non-FL(MLP)</td>
    <td class="tg-nrix">1e-4</td>
    <td class="tg-nrix">1e-4</td>
    <td class="tg-nrix">1e-4</td>
    <td class="tg-nrix">1e-4</td>
  </tr>
  <tr>
    <td class="tg-nrix">non-FL(bi-LSTM)</td>
    <td class="tg-nrix">1e-4</td>
    <td class="tg-nrix">1e-2</td>
    <td class="tg-nrix">1e-4</td>
    <td class="tg-nrix">1e-4</td>
  </tr>
  <tr>
    <td class="tg-nrix">FedAvg</td>
    <td class="tg-nrix">1e-3</td>
    <td class="tg-nrix">1e-2</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-3</td>
  </tr>
  <tr>
    <td class="tg-nrix">FedAdam</td>
    <td class="tg-nrix">1e-4.5</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-4.5</td>
    <td class="tg-nrix">1e-5</td>
  </tr>
  <tr>
    <td class="tg-nrix">FedProx</td>
    <td class="tg-nrix">1e-3</td>
    <td class="tg-nrix">1e-2</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-3</td>
  </tr>
  <tr>
    <td class="tg-nrix">MOON</td>
    <td class="tg-nrix">1e-3</td>
    <td class="tg-nrix">1e-2</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-3</td>
  </tr>
  <tr>
    <td class="tg-nrix">FedAwS</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-2</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-5</td>
  </tr>
  <tr>
    <td class="tg-nrix">TurboSVM-FL</td>
    <td class="tg-nrix">1e-3</td>
    <td class="tg-nrix">1e-3.5</td>
    <td class="tg-nrix">1e-3</td>
    <td class="tg-nrix">1e-5</td>
  </tr>
</tbody>
</table>

While running 5-fold cross-validation on the training set, for the Colorado and Korea datasets, it can happen that all users in some fold share the same single class label regarding glass wearing behavior. Therefore, we manually exchanged a small portion of samples across folds. The statistics tables above show a thorough analysis of the train and test sets for all four datasets. The exact client division of each dataset can be found in our GitHub repository.
