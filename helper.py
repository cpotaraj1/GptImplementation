
"""
Add custom code to save model during the training process

"""

import torch

# Define a variable to keep track of the best validation loss
best_val_loss = float('inf')

# Define your model, optimizer, loss function, and data loaders

for epoch in range(num_epochs):
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass, backward pass, optimization
        
    # Validation loop
    with torch.no_grad():
        val_loss = 0
        for data, target in val_loader:
            # Compute validation loss
        
    # Check if validation loss improved
    if val_loss < best_val_loss:
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

# Load the best model
best_model = YourModelClass()
best_model.load_state_dict(torch.load('best_model.pth'))


# In[ ]:


"""
extractive approach and the TextRank algorithm. In this approach, we'll rank the sentences in the input text based on their importance
and select the top-ranked sentences to form the summary.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = [[word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words] for sentence in sentences]

    return tokenized_sentences

def calculate_similarity_matrix(sentences):
    # Create a matrix to store the similarity scores between sentences
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Calculate similarity scores between sentences using cosine similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                vector1 = np.mean(word_embeddings[sentences[i]], axis=0)
                vector2 = np.mean(word_embeddings[sentences[j]], axis=0)
                similarity_matrix[i][j] = cosine_similarity([vector1], [vector2])[0,0]

    return similarity_matrix

def textrank_summary(text, num_sentences=3):
    # Preprocess the text
    tokenized_sentences = preprocess_text(text)

    # Calculate word embeddings for each word in the text
    word_embeddings = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in word_embeddings:
                word_embeddings[word] = np.random.rand(300,)  # Random word embeddings for demonstration purposes

    # Calculate similarity matrix between sentences
    similarity_matrix = calculate_similarity_matrix(tokenized_sentences)

    # Calculate PageRank scores using the similarity matrix
    scores = np.zeros(len(tokenized_sentences))
    for i in range(len(tokenized_sentences)):
        scores[i] = 1.0 / len(tokenized_sentences)
    damping_factor = 0.85  # Damping factor for PageRank
    for _ in range(50):  # Iterate PageRank calculation for a fixed number of iterations
        new_scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix.T, scores)
        if np.allclose(new_scores, scores, atol=0.001):
            break
        scores = new_scores

    # Sort the sentences based on their PageRank scores
    ranked_sentences_indices = np.argsort(-scores)
    
    # Select the top-ranked sentences to form the summary
    summary = [tokenized_sentences[i] for i in ranked_sentences_indices[:num_sentences]]
    return summary

# Example usage
"""
Text summarization is the process of distilling the most important information from a source (or sources) 
to produce an abridged version for a particular user (or users) and task (or tasks). 
There are generally two main approaches to text summarization: extractive and abstractive. 
Extractive summarization involves selecting a subset of phrases, sentences, or paragraphs from the source text to create the summary. 
In contrast, abstractive summarization involves generating new phrases, sentences, or paragraphs that capture the key information 
from the source text but may not appear verbatim in the original text. 
In this example, we'll focus on extractive text summarization using the TextRank algorithm.
"""
input_text = ""


summary = textrank_summary(input_text)
print("Summary:")
for sentence in summary:
    print(' '.join(sentence))


"""
This implementation uses a simple version of the TextRank algorithm to rank the sentences based on their 
importance and select the top-ranked sentences to form the summary. Please note that for better performance, 
you may want to replace the random word embeddings with pre-trained word embeddings like Word2Vec or GloVe and 
fine-tune the similarity calculation accordingly. Additionally, you may need to refine the preprocessing steps 
and adjust parameters based on your specific text data and requirements.
"""
# ___________________________________________________________________________________________________________________________________________
"""
This is TFIDF approach to summarize the notes

TF-IDF calculates the importance of each word in a document based on its frequency 
in the document and its rarity across all documents. 
Then, we select the most important sentences based on the TF-IDF scores of their words to form the summary.


"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample notes
notes = """
This is a sample note. It contains multiple sentences. We want to summarize it.
"""

# Tokenize the notes into sentences
sentences = sent_tokenize(notes)

# Tokenize each sentence into words and remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
tokenized_sentences = [[word for word in words if word not in stop_words] for words in tokenized_sentences]

# Convert tokenized sentences back to strings
preprocessed_sentences = [' '.join(words) for words in tokenized_sentences]

# Calculate TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

# Get TF-IDF scores for each word
feature_names = vectorizer.get_feature_names_out()
word_scores = dict(zip(feature_names, vectorizer.idf_))

# Calculate sentence scores based on TF-IDF scores of words
sentence_scores = [sum(word_scores[word] for word in words) for words in tokenized_sentences]

# Select top sentences based on scores to form the summary
summary_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)[:2]]

# Combine summary sentences into a summary paragraph
summary = ' '.join(summary_sentences)

print("Summary:")
print(summary)