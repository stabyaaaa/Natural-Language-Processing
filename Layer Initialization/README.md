# Task 1: Student Layer Initilization
Student Layer Initialization - Based on case-studies/distilBERT.ipynb, modify as follows:
1) Initialize the top K layers {1,2,3,4,5,6} from 12-layers teacher to 6-layers student. (1 points) ✅
2) Initialize the bottom K layers {7,8,9,10,11,12} from 12-layers teacher to 6-layers student. (1 points) ✅
3) Initialize the odd layers {1,3,5,7,8,9,11} from 12-layers teacher to 6-layers student. (1 points) ✅

# Task 2: Evaluation and Analysis
## 2.1 Perform a detailed evaluation of your distilled student model, analyzing the impact of the initiallayer selection (top K layers, bottom K layers, odd layers) on its performance. (1 points)

|Student Layer  | Training Loss | Validation Loss | Validation Accuracy |
|---------------|---------------|-----------------|---------------------|
| Top-K  Layer  | 0.2754        | 0.7990          | 0.6660              |
| Bottom-K Layer| 0.2660        | 0.7990          | 0.6660              |
| Odd Layer     | 0.262         | 0.7990          | 0.6660              |
| Even Layer    | 0.2662        | 0.7990          | 0.6660              |

Four different distilled student models were trained with different initial layer selections: Top-K Layer, Bottom-K Layer, Odd Layer, and Even Layer. Each model's training losses, evaluation losses, and evaluation accuracy are calculated.

From the above table, it seems that the choice of the initial layer selection does not significantly impact the performance of the distilled student model. All four models have similar evaluation losses and evaluation accuracy. Other factors such as the architecture of the student model, the complexity of the task, and the dataset used for training could also influence the results.

## 2.2 Discuss any limitations or challenges encountered during the implementation of student distillation, specifically focusing on the analysis of how the initial layer selection affects the overall performance. Propose potential improvements or modifications to address these challenges. (1 points)

One major challenge was determining the optimal initial layer selection strategy. Initially, I experimented with different approaches such as selecting the top-K layers, bottom-K layers, odd layers, and even layers. However, it was difficult to ascertain which strategy would yield the best results without extensive experimentation.

#### To address this issue:
By conducting more comprehensive analyses, exploring alternative strategies, leveraging ensemble methods, and fine-tuning hyperparameters, it's possible to overcome these limitations and improve the effectiveness of student distillation.


