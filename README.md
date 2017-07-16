# Digit Recognizer #
  
This repository contains the code to solve Kaggle contest "digit recognizer".  
Using **neural network** and **PCA dimension reduction**.

### Contributors ###
Chen-Hsi (Sky) Huang (github.com/skyhuang1208)   
Louis Yang (github.com/louis925)

### Achievement ###
An 98.071% accuracy was achieved (verified by kaggle).

### How it works ###
1. Read in training data and testing data (42000, 28000 for this contest).
2. (Optional) Using PCA dimension reduction to reduce data size (use size with 95% explained variance).
3. (a) Tuning parameters (layers, neurons, learning rate, tolerance, etc.) 
3. (b) Train feed-forward artificial neural network using training data.
4. Output predictions on digits for testing data set.

### How to Use ###
First enter parameters on the top of digit_recognizer.py
Use Python3 to execute digit_recognizer.py such as:
python3 digit_recognizer.py

