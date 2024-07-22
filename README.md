# Movie-TV-Show-Review-Classification

---
title: "Movie and TV Show Review Classification with Logistic Regression"
---

## Task

In a fictional data science competition hosted by Marvin from Blockbusta Videoz on the Fraggle platform, the task was to classify texts into three categories: not a movie/TV show review, a positive review, or a negative review. The primary metric for evaluation is the macro F1 score, which is used to assess the model’s accuracy across the three classes by averaging the F1 scores.

### Dataset

The dataset is provided in CSV format and contains 70,311 entries, divided among three categories with the following distribution: 32,065 entries not related to movie/TV show reviews (Label 0), 19,276 positive reviews (Label 1), and 18,970 negative reviews (Label 2). Each entry includes an 'ID', a unique identifier, 'TEXT', the actual text content of the document, and 'LABEL', to classify each document. 

### Challenges of the Task

**Imbalanced Class Distribution**: Disparity in the number of examples for each class could lead to model bias towards the majority class.

**Variable Text Lengths**: Texts vary widely in length, from very short to extremely long, requiring the model to effectively process diverse text sizes.

**Complex Language and Sentiment**: The data includes complex language features like sarcasm and mixed emotions, which is challenging for a model to interpret accurately.

**High Vocabulary Diversity**: A large and diverse vocabulary increases model complexity.

**Varying Sentiment Scores**: The data features a broad spectrum of sentiment levels, necessitating precise modeling to effectively utilize sentiment in classification tasks.

### Related Task

**Sentiment Analysis**: Closely related to review classification, sentiment analysis involves determining the sentiment expressed in a piece of text, classified as positive, negative, or neutral.

## Approach

My approach to classifying movie and TV show reviews involves a multi-faceted machine learning pipeline. Here’s a breakdown of the methodology:

**Data Preprocessing**: 

Initial steps involve cleaning the data, handling missing values, and standardizing text format to ensure uniformity. This stage is crucial for the effective application of NLP techniques.

**Feature Engineering**:
- **TF-IDF Vectorization**: This method was employed to transform the text into a format suitable for model input, emphasizing important words and phrases.
- **Sentiment Analysis**: By integrating NLTK’s SentimentIntensityAnalyzer, I generate sentiment scores for each review, adding an extra layer of data that helps in distinguishing between positive and negative sentiments.

**Model Selection and Training**:
- **Logistic Regression with saga solver**: Chosen for its efficiency and effectiveness in multi-class classification problems, logistic regression with saga solver provides a good baseline for this task.
- **Sentiment Feature Integration**: Including sentiment scores as features allows the model to leverage emotional content in the reviews, which is especially helpful for distinguishing between positive and negative reviews.

**Model Evaluation**:

I use stratified cross-validation to assess the model’s performance, ensuring its reliability and robustness. The macro F1 score, as evaluation metric, helps balance the precision and recall across the unbalanced classes.


## Results

**Model Performance**: 

To ensure the robustness of my approach, I employed stratified 5-fold cross-validation, achieving a mean F1 score of 0.925 (± 0.001). This method validates the model's performance across different subsets of the dataset, providing a more accurate and generalizable assessment of the model's effectiveness.

## Code

The entire process, from data preprocessing to model evaluation, is documented in a Jupyter Notebook.

This repository provides the script and datasets used, making it easy for others to replicate the results or use the model as a baseline for further experiments.

### Reproducing Results

To reproduce the results, follow these steps:

##### Step 1: Clone the Repository

##### Step 2: Create a New Virtual Environment

Set up a new virtual environment to manage dependencies:

##### For Windows:

Navigate to the project directory in the command prompt, then run:

```bash
python -m venv venv
.\venv\Scripts\activate
```
##### For macOS and Linux:

Navigate to the project directory in the terminal, then run:

```bash
python3 -m venv venv
source venv/bin/activate
```
##### Step 3: Install Jupyter Notebook and Dependencies

With the virtual environment activated, install Jupyter Notebook and other required packages:

```bash
pip install notebook
pip install -r requirements.txt
```
##### Step 4: Start Jupyter Notebook

Launch Jupyter Notebook by running:

```bash
jupyter notebook
```

##### Step 5: Run classifier.ipynb

Open the classifier.ipynb file in Jupyter Notebook. This notebook includes all the necessary steps to reproduce the results, with comments explaining each part of the process.

## Future Improvements

**Hyperparameter Tuning**: Further tuning of the model's hyperparameters using grid search or randomized search could yield better results.

**Advanced Sentiment Analysis Techniques**: Many of the high confidence errors involved incorrect sentiment interpretation, especially in cases of complex reviews that express nuanced sentiments or mixed feelings. For example, the review where the film 'Nineteen Eighty-Four' is described as having 'some very good ideas' and being 'done well' but also criticized for being 'very slow' and not fully conveying its ideas, illustrates the challenge of classifying mixed sentiments. Integrating more sophisticated sentiment analysis tools, such as BERT-based sentiment classifiers that can understand context more deeply, could help in better capturing these subtleties and reducing misclassifications.

**Contextual Understanding and Text Complexity**: The error analysis highlighted cases where the text involved complex expressions, sarcasm, or references that are not straightforward to interpret. For instance, a review sarcastically remarked, 'If there were an EPA for film, then this movie would get their most sincere approval. If we all recycled our "stuff" to this degree, we'd never run out of anything,' which was misclassified as positive. This error likely occurred due to the model's inability to recognize the sarcastic use of 'sincere approval' to criticize the film’s lack of originality. Utilizing natural language understanding models like BERT or RoBERTa could improve the model’s ability to understand and accurately classify texts based on deeper contextual cues.

## Conclusion

The approach provides a good baseline for this task. Future improvements as stated in the above section could further enhance the model's accuracy and utility, making it an even more effective tool for classifying movie/TV show reviews.
