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

## Web Application
To access the web application for the Sentence Embedding, you can run the `main.py` file. Once the Flask server is up and running, you can access the application by navigating to `localhost:5000` in your web browser.

Make sure to update the necessary configurations, such as host and port settings, in the main.py file if you want to run the application on a different address or port.

You can start the Flask server by executing the following command in your terminal:
`python main.py`
