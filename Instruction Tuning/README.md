# 1. Instruction Tuning
In this assignment, users will be able to instruct the model based upon the input

#### 1.1 Dataset
DAtaset for this task has been taken from github.
Link : https://github.com/gururise/AlpacaDataCleaned


# 2. Demo
![Alpaca demo](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/assets/35591848/1d5a5eb8-ab2a-402d-9907-aa874849ee8c)


# 3. Performance Analysis:
| Epoch      | Training Loss |  Validation Loss | 
|------------|---------------|------------------|
| 1          | 2.6654        |  2.2666          | 
| 2          | 2.4404        | 2.2646           | 
| 3          | 1.9002        |2.2796            |


The final report for training and validation loss is as follows:
Training Loss Trend: Decreasing consistently across epochs, indicating learning.
Validation Loss Trend: Initially decreases but slightly increases in the final epoch.
Overfitting: Possible, indicated by the increasing validation loss in the final epoch.
Model Performance: Achieves low training loss but may not generalize optimally to unseen data.
Recommendation: Further investigation into architecture, hyperparameters, and regularization techniques may be needed.

# 4. Web Application
To access the web application for the AIT GPT, you can run the `main.py` file. Once the Flask server is up and running, you can access the application by navigating to `localhost:5000` in your web browser.

Make sure to update the necessary configurations, such as host and port settings, in the main.py file if you want to run the application on a different address or port.

You can start the Flask server by executing the following command in your terminal:
`python main.py`
