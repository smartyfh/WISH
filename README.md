# WISH

Few-Bits Semantic Hashing

Code in Python for the paper "Unsupervised Few-Bits Semantic Hashing with Implicit Topics Modeling" [Findings of EMNLP 2020].

### Train

1. Data Preprocessing

    ```
    python3 datapreprocessing.py
    ```
    
2. Training

   If the datasets have multiple labels for a single sample, run

   ```
     python3 train_WISH.py
   ```
    
   Otherwise, run

    ```
      python3 train_WISH.py --single_label_flag
    ```
    
