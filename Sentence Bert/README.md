# 1. Sentence Embedding with Bert
In this assignment, users will check the similarity betweeen two sentences. 
#### 1.1 Dataset
Dataset for this assisgnment has been taken from Hugging face which is called sample wiki. You can download it from: https://huggingface.co/datasets/embedding-data/simple-wiki


# 2. Demo
![Sentence Embedding](https://github.com/stabyaaaa/Natural-Language-Processing/assets/35591848/e72444fa-043e-48f5-af3a-b1fb9ca882f4)

# 3. Model Comparision

| Model            | Training Loss | Average Evaluation Cosine Similarity |
|------------------|---------------|--------------------------------------|
| Cusstom Model    | 17.300081     | 0.9788                               |
| Pretrained Model | 11.925        | 0.9999998                            |


# 4. Difficulty faced
    Since my GPU memory is only 4GB, I faced a lot a issue regarding CUDA memory getting out of space. EVen though CUDA was available, training took more than 12-13 minutes for one epoch. I reduced the batch size from 8 / 4 to 2, then this problem was solved. But it was time consuming..

# 5. Brief Model comparision and overview
    Custom model couldn't perform better than the pretrained model which is obvious.
    The pretrained model has a lower training loss (11.925) compared to the custom model (17.300081). This suggests that the pretrained model is better at learning from the training data and making accurate predictions.

    The pretrained model has a higher average evaluation cosine similarity (0.9999998) compared to the custom model (0.9788). This indicates that the pretrained model's embeddings or representations are more similar to each other on average, suggesting a higher quality of semantic understanding or feature representation.

    Pretrained models are often trained on large-scale datasets and have access to vast amounts of pre-existing knowledge, allowing them to capture intricate patterns and relationships in the data.
    Custom models, on the other hand, are typically trained on smaller or domain-specific datasets, which may limit their ability to generalize to unseen data or capture complex patterns present in the data.

    Overall, the comparison highlights the benefits of using pretrained models, particularly in scenarios where large amounts of data are available and where high performance is crucial. However, it's essential to consider factors such as computational resources, task-specific requirements, and the availability of labeled data when choosing between pretrained models and custom models for a particular application.

# 6.mpact of hyperparameter choices on the model’s performance
    Hyperparameter tuning involves finding the optimal configuration for parameters like learning rate, batch size, and regularization strength, crucially impacting a model's ability to generalize and achieve desired performance levels.

# 7. Web Application
    To access the web application for the Sentence Embedding, you can run the `main.py` file. Once the Flask server is up and running, you can access the application by navigating to `localhost:5000` in your web browser.

    Make sure to update the necessary configurations, such as host and port settings, in the main.py file if you want to run the application on a different address or port.

    You can start the Flask server by executing the following command in your terminal:
    `python main.py`
