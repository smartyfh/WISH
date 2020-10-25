# WISH

Code in Python for the paper "Unsupervised Few-Bits Semantic Hashing with Implicit Topics Modeling".

### Train
1. Data Preprocessing

    ```
    python3 datapreprocessing.py
    ```
    
2. Training

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If the datasets have multiple labels for a single sample, run

     python3 train_WISH.py
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Otherwise, run

 
     python3 train_WISH.py --single_label_flag

