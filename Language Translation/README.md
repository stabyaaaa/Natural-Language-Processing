# Language Translation Model by Stabya Acharya st124150

### Datasets 
#### 
Dataset for this model has been used using `opus100` available through the Hugging Face `datasets` library.
demo : ![demofinal](https://github.com/stabyaaaa/Natural-Language-Processing/assets/35591848/39d6089d-c1fd-4236-bb1f-f7f91ebd698e)

### PREPROCESSING STEPS
Random Sampling: Select a subset of training data.
Tokenization: Break text into smaller units.
Normalization: Standardize text format.
Word Segmentation: Segment text for languages without spaces.
Mapping Tokenization Function: Apply tokenization function to the dataset.

`numpy.random` for random sampling.
`torchtext.data.utils` for tokenization.
`nepalitokenizers for` Nepali tokenization.


### Task 2: Experiment with Attentions Mechanisms
1. General Attention
2. Multiplicative Attention
3. Additive Attention


| Attention Variant | Training Loss | Training PPL | Validation Loss | Validation PPL   |
|-------------------|---------------|--------------|------------------|-----------------|
| General           | 6.118         | 454.087      | 5.43             |  228.504        |
| Multiplicative    | 6.84          | 933.699      | 5.99             | 398.01          |
| Additive          | 6.62          | 753.33       | 5.81             | 332.82          |

### Comparision Betweeen Attentions | Training


Based on the provided results:

General Attention: Achieved the lowest training and validation losses, resulting in the lowest perplexity values on both training and validation sets. This suggests that the general attention mechanism performed the best overall in capturing the dependencies between source and target language sequences during translation.

Multiplicative Attention: Demonstrated higher training and validation losses compared to the general attention mechanism. The higher perplexity values indicate that this attention mechanism may not have effectively captured the relationships between source and target language tokens as efficiently as the general attention mechanism.

Additive Attention: Showed intermediate results between general and multiplicative attention. While it performed better than multiplicative attention, it still had slightly higher training and validation losses than the general attention mechanism.

In summary, the general attention mechanism appears to be the most effective for translating between the native language and English. Its ability to capture dependencies between language sequences better than the other mechanisms results in lower perplexity values, indicating a higher level of fluency and coherence in translations. However, further analysis, including qualitative evaluation and tuning, may be necessary to fully understand the effectiveness of each attention mechanism in the translation task.

### Plots of Attention models
1. General attention: 
2. Multiplicative Attention:
3. Addtitive Attention:

## Web Application
To access the web application for the translation model, you can run the main.py file. Once the Flask server is up and running, you can access the application by navigating to localhost:5000 in your web browser.

Make sure to update the necessary configurations, such as host and port settings, in the main.py file if you want to run the application on a different address or port.

You can start the Flask server by executing the following command in your terminal:
`python main.py`


