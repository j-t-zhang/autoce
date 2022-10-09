# AutoCE

## Data preparation
```python
cd data_preparation
python main.py
```

## AutoCE training and recommendation
### Baselines proposed by us
- MLP-based
  ```python
  cd advisor_baselines
  python main_classify.py
  ```
- Knn-based
  ```python
  cd advisor_baselines
  python Knn.py
  ```
- Ablation of DML
  ```python
  cd advisor_baselines
  python main_regression.py
  ```
### AutoCE
  ```python
  cd contrast_train
  python contrast.py
  python incremental_train.py
  python KNN.py
  ```
  - A trained model is in `contrast_train/model/embedding.pth`

### Integrate into end-to-end systems
- First, refer to the code of https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark to install a modified version of postgresSQL based on docker implementation.
  ```python
  cd e2e_time
  python gen_sub_queries.py
  # Estimate the cardinalities of the subqueries and save the result as [method].txt
  python e2e_time.py
  ```
