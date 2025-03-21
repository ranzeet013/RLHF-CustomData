import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

class Judger:
    """
    A class to evaluate and score model outputs based on multiple criteria.
    """

    def __init__(self):
        """
        Initialize the Judger class with a pre-trained sentence embedding model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def check_correctness(self, output):
        """
        Placeholder function to check factual correctness.
        In a real implementation, use a fact-checking API or knowledge base.
        """
        return 1.0  

    def calculate_relevance(self, input_text, output_text):
        """
        Calculate relevance as the cosine similarity between input and output embeddings.
        """
        input_embedding = self.model.encode([input_text])
        output_embedding = self.model.encode([output_text])
        relevance_score = cosine_similarity(input_embedding, output_embedding)[0][0]
        return relevance_score

    def calculate_clarity(self, output_text):
        """
        Placeholder function to calculate clarity.
        In a real implementation, use a readability score or grammar-checking tool.
        """
        # Simple heuristic: shorter sentences are clearer
        word_count = len(output_text.split())
        clarity_score = 1 / (1 + math.exp(-0.1 * (50 - word_count))) 
        return clarity_score

    def calculate_coherence(self, output_text):
        """
        Placeholder function to calculate coherence.
        In a real implementation, use a language model to compute perplexity.
        """
        return 1.0  

    def calculate_usefulness(self, output_text):
        """
        Placeholder function to calculate usefulness.
        In a real implementation, use human annotations or predefined utility scores.
        """
        return 1.0  

    def calculate_conciseness(self, output_text):
        """
        Calculate conciseness as the inverse of word count.
        """
        word_count = len(output_text.split())
        conciseness_score = 1 / (1 + word_count)  
        return conciseness_score

    def calculate_engagement(self, output_text):
        """
        Placeholder function to calculate engagement.
        In a real implementation, use sentiment analysis or engagement prediction models.
        """
        return 1.0  # Assume all outputs are engaging for now

    def calculate_preference_score(self, input_text, output_text):
        """
        Calculate the overall preference score for an output based on multiple criteria.
        """
        # Calculate individual scores
        correctness = self.check_correctness(output_text)
        relevance = self.calculate_relevance(input_text, output_text)
        clarity = self.calculate_clarity(output_text)
        coherence = self.calculate_coherence(output_text)
        usefulness = self.calculate_usefulness(output_text)
        conciseness = self.calculate_conciseness(output_text)
        engagement = self.calculate_engagement(output_text)

        # Define weights for each criterion
        weights = {
            "correctness": 0.4,
            "relevance": 0.3,
            "clarity": 0.1,
            "coherence": 0.1,
            "usefulness": 0.05,
            "conciseness": 0.03,
            "engagement": 0.02
        }

        # Calculate weighted sum
        preference_score = (
            weights["correctness"] * correctness +
            weights["relevance"] * relevance +
            weights["clarity"] * clarity +
            weights["coherence"] * coherence +
            weights["usefulness"] * usefulness +
            weights["conciseness"] * conciseness +
            weights["engagement"] * engagement
        )

        return preference_score