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





