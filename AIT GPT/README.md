# 1. AIT GPT
In this assignment, users will check the similarity betweeen two sentences. 
#### 1.1 Dataset
DAtaset for this task has been taken from various internet sources such as wikipedia which has been converted tto pdf file.


# 2. Demo
![Sentence Embedding](https://github.com/stabyaaaa/Natural-Language-Processing/assets/35591848/e72444fa-043e-48f5-af3a-b1fb9ca882f4)
 and regularization strength, crucially impacting a model's ability to generalize and achieve desired performance levels.

# 3. Performance Analysis:

1. The model demonstrates satisfactory performance in retrieving information from the dataset. It accurately identifies and presents relevant information based on the provided prompts.
2. Despite the limited GPU resources (4GB), the model efficiently processes requests and retrieves information within reasonable time frames, showcasing its effectiveness in handling tasks even with resource constraints.

# 4. Addressing Unrelated Information Issues:

Implementing a relevance scoring mechanism could help mitigate the issue of the model providing unrelated information. By assigning scores to potential responses based on their relevance to the prompt, the model can prioritize and present the most relevant information to the user.
Fine-tuning the model on domain-specific data, such as educational resources relevant to Asian Institute of Technology (AIT), can improve its understanding of context and reduce the likelihood of generating irrelevant responses. This approach ensures that the model is trained on data closely aligned with the user's needs, thereby enhancing the quality of retrieved information.
Introducing context-aware filtering techniques, such as contextual embeddings or attention mechanisms, can enhance the model's ability to discern the context of the prompt and generate responses that are more closely related to the given topic. By focusing on contextual cues within the prompt, the model can better filter out irrelevant information and provide more accurate and targeted responses.

The dataset used for training the model primarily consists of Wikipedia articles relevant to a wide range of topics. While this provides a diverse knowledge base, it may also introduce noise and unrelated information, leading to occasional instances of the model providing irrelevant responses.

# 7. Web Application
To access the web application for the AIT GPT, you can run the `main.py` file. Once the Flask server is up and running, you can access the application by navigating to `localhost:5000` in your web browser.

Make sure to update the necessary configurations, such as host and port settings, in the main.py file if you want to run the application on a different address or port.

You can start the Flask server by executing the following command in your terminal:
`python main.py`
