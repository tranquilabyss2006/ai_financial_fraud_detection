"""
NLP Features Module
Implements natural language processing features for fraud detection
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import warnings
import logging
from typing import Dict, List, Tuple, Union

from fraud_detection_engine.utils.api_utils import is_api_available

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NLPFeatures:
    """
    Class for extracting NLP features from transaction data
    Implements techniques like sentiment analysis, topic modeling, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize NLPFeatures
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.word2vec_model = None
        self.doc2vec_model = None
        self.feature_names = []
        self.fitted = False
        
        # Fraud-related keywords
        self.fraud_keywords = [
            'urgent', 'immediately', 'asap', 'hurry', 'quick', 'fast',
            'secret', 'confidential', 'private', 'hidden', 'discreet',
            'suspicious', 'unusual', 'strange', 'odd', 'weird',
            'illegal', 'fraud', 'scam', 'fake', 'counterfeit',
            'money', 'cash', 'payment', 'transfer', 'wire',
            'overseas', 'foreign', 'international', 'abroad',
            'inheritance', 'lottery', 'prize', 'winner', 'claim',
            'verify', 'confirm', 'update', 'account', 'information',
            'click', 'link', 'website', 'login', 'password',
            'bank', 'check', 'routing', 'account', 'number'
        ]
        
        # Suspicious patterns
        self.suspicious_patterns = [
            r'\$\d+,\d+\.\d{2}',  # Money format
            r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',  # Credit card pattern
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email pattern
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL pattern
            r'\b\d{10,}\b',  # Long numbers (could be account numbers)
            r'[A-Z]{2,}',  # All caps words
            r'\d+\.\d+\.\d+\.\d+',  # IP address pattern
        ]
    
    def extract_features(self, df):
        """
        Extract all NLP features from the dataframe
        
        Args:
            df (DataFrame): Input transaction data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Extract different types of NLP features
            result_df = self._extract_basic_text_features(result_df)
            result_df = self._extract_sentiment_features(result_df)
            result_df = self._extract_keyword_features(result_df)
            result_df = self._extract_pattern_features(result_df)
            result_df = self._extract_topic_features(result_df)
            result_df = self._extract_embedding_features(result_df)
            
            # Store feature names
            self.feature_names = [col for col in result_df.columns if col not in df.columns]
            
            logger.info(f"Extracted {len(self.feature_names)} NLP features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting NLP features: {str(e)}")
            raise
    
    def _extract_basic_text_features(self, df):
        """
        Extract basic text features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with basic text features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            for col in text_columns:
                if col in df.columns:
                    # Character count
                    result_df[f'{col}_char_count'] = df[col].fillna('').astype(str).apply(len)
                    
                    # Word count
                    result_df[f'{col}_word_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(word_tokenize(x)) if x else 0
                    )
                    
                    # Sentence count
                    result_df[f'{col}_sentence_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(sent_tokenize(x)) if x else 0
                    )
                    
                    # Average word length
                    result_df[f'{col}_avg_word_length'] = df[col].fillna('').astype(str).apply(
                        lambda x: np.mean([len(word) for word in word_tokenize(x)]) if word_tokenize(x) else 0
                    )
                    
                    # Average sentence length
                    result_df[f'{col}_avg_sentence_length'] = df[col].fillna('').astype(str).apply(
                        lambda x: np.mean([len(word_tokenize(sent)) for sent in sent_tokenize(x)]) if sent_tokenize(x) else 0
                    )
                    
                    # Punctuation count
                    result_df[f'{col}_punctuation_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: sum(1 for char in x if char in string.punctuation)
                    )
                    
                    # Uppercase word count
                    result_df[f'{col}_uppercase_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: sum(1 for word in word_tokenize(x) if word.isupper() and len(word) > 1)
                    )
                    
                    # Digit count
                    result_df[f'{col}_digit_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: sum(1 for char in x if char.isdigit())
                    )
                    
                    # Unique word count
                    result_df[f'{col}_unique_word_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(set(word_tokenize(x.lower()))) if x else 0
                    )
                    
                    # Lexical diversity (unique words / total words)
                    result_df[f'{col}_lexical_diversity'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(set(word_tokenize(x.lower()))) / len(word_tokenize(x)) if word_tokenize(x) else 0
                    )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting basic text features: {str(e)}")
            return df
    
    def _extract_sentiment_features(self, df):
        """
        Extract sentiment analysis features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with sentiment features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            for col in text_columns:
                if col in df.columns:
                    # VADER sentiment scores
                    sentiment_scores = df[col].fillna('').astype(str).apply(
                        lambda x: self.sia.polarity_scores(x) if x else {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
                    )
                    
                    # Extract individual scores
                    result_df[f'{col}_sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
                    result_df[f'{col}_sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
                    result_df[f'{col}_sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
                    result_df[f'{col}_sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
                    
                    # TextBlob sentiment
                    textblob_sentiment = df[col].fillna('').astype(str).apply(
                        lambda x: TextBlob(x).sentiment if x else TextBlob('').sentiment
                    )
                    
                    result_df[f'{col}_textblob_polarity'] = textblob_sentiment.apply(lambda x: x.polarity)
                    result_df[f'{col}_textblob_subjectivity'] = textblob_sentiment.apply(lambda x: x.subjectivity)
                    
                    # Emotion indicators
                    result_df[f'{col}_is_negative'] = (result_df[f'{col}_sentiment_compound'] < -0.05).astype(int)
                    result_df[f'{col}_is_positive'] = (result_df[f'{col}_sentiment_compound'] > 0.05).astype(int)
                    result_df[f'{col}_is_neutral'] = (
                        (result_df[f'{col}_sentiment_compound'] >= -0.05) & 
                        (result_df[f'{col}_sentiment_compound'] <= 0.05)
                    ).astype(int)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {str(e)}")
            return df
    
    def _extract_keyword_features(self, df):
        """
        Extract keyword-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with keyword features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            for col in text_columns:
                if col in df.columns:
                    # Preprocess text
                    processed_text = df[col].fillna('').astype(str).apply(self._preprocess_text)
                    
                    # Count fraud keywords
                    fraud_keyword_counts = processed_text.apply(
                        lambda x: sum(1 for word in x if word in self.fraud_keywords)
                    )
                    result_df[f'{col}_fraud_keyword_count'] = fraud_keyword_counts
                    
                    # Flag if any fraud keywords present
                    result_df[f'{col}_has_fraud_keywords'] = (fraud_keyword_counts > 0).astype(int)
                    
                    # Count specific keyword categories
                    urgency_keywords = ['urgent', 'immediately', 'asap', 'hurry', 'quick', 'fast']
                    secrecy_keywords = ['secret', 'confidential', 'private', 'hidden', 'discreet']
                    money_keywords = ['money', 'cash', 'payment', 'transfer', 'wire']
                    
                    result_df[f'{col}_urgency_keyword_count'] = processed_text.apply(
                        lambda x: sum(1 for word in x if word in urgency_keywords)
                    )
                    
                    result_df[f'{col}_secrecy_keyword_count'] = processed_text.apply(
                        lambda x: sum(1 for word in x if word in secrecy_keywords)
                    )
                    
                    result_df[f'{col}_money_keyword_count'] = processed_text.apply(
                        lambda x: sum(1 for word in x if word in money_keywords)
                    )
                    
                    # Calculate keyword density
                    result_df[f'{col}_fraud_keyword_density'] = fraud_keyword_counts / (
                        df[col].fillna('').astype(str).apply(lambda x: len(word_tokenize(x)) if x else 1)
                    )
                    
                    # TF-IDF for fraud keywords
                    if not self.fitted:
                        # Fit TF-IDF vectorizer
                        self.tfidf_vectorizer = TfidfVectorizer(
                            vocabulary=self.fraud_keywords,
                            ngram_range=(1, 2),
                            max_features=100
                        )
                        
                        # Fit on all text
                        all_text = pd.concat([df[col].fillna('').astype(str) for col in text_columns if col in df.columns])
                        self.tfidf_vectorizer.fit(all_text)
                    
                    # Transform text
                    tfidf_features = self.tfidf_vectorizer.transform(df[col].fillna('').astype(str))
                    
                    # Add top TF-IDF features
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    for i, feature in enumerate(feature_names[:10]):  # Top 10 features
                        result_df[f'{col}_tfidf_{feature}'] = tfidf_features[:, i].toarray().flatten()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting keyword features: {str(e)}")
            return df
    
    def _extract_pattern_features(self, df):
        """
        Extract pattern-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with pattern features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            for col in text_columns:
                if col in df.columns:
                    # Count suspicious patterns
                    pattern_counts = []
                    for text in df[col].fillna('').astype(str):
                        count = 0
                        for pattern in self.suspicious_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            count += len(matches)
                        pattern_counts.append(count)
                    
                    result_df[f'{col}_suspicious_pattern_count'] = pattern_counts
                    
                    # Flag if any suspicious patterns present
                    result_df[f'{col}_has_suspicious_patterns'] = (np.array(pattern_counts) > 0).astype(int)
                    
                    # Specific pattern counts
                    money_pattern = r'\$\d+,\d+\.\d{2}'
                    credit_card_pattern = r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'
                    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
                    email_pattern = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
                    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    
                    result_df[f'{col}_money_pattern_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(re.findall(money_pattern, x))
                    )
                    
                    result_df[f'{col}_credit_card_pattern_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(re.findall(credit_card_pattern, x))
                    )
                    
                    result_df[f'{col}_ssn_pattern_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(re.findall(ssn_pattern, x))
                    )
                    
                    result_df[f'{col}_email_pattern_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(re.findall(email_pattern, x))
                    )
                    
                    result_df[f'{col}_url_pattern_count'] = df[col].fillna('').astype(str).apply(
                        lambda x: len(re.findall(url_pattern, x))
                    )
                    
                    # Check for excessive punctuation
                    result_df[f'{col}_excessive_punctuation'] = (
                        df[col].fillna('').astype(str).apply(
                            lambda x: 1 if sum(1 for char in x if char in string.punctuation) / len(x) > 0.3 else 0
                        )
                    )
                    
                    # Check for excessive capitalization
                    result_df[f'{col}_excessive_capitalization'] = (
                        df[col].fillna('').astype(str).apply(
                            lambda x: 1 if sum(1 for char in x if char.isupper()) / len(x) > 0.5 else 0
                        )
                    )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {str(e)}")
            return df
    
    def _extract_topic_features(self, df):
        """
        Extract topic modeling features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with topic features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            # Combine text from all columns
            all_text = []
            for col in text_columns:
                if col in df.columns:
                    all_text.extend(df[col].fillna('').astype(str).tolist())
            
            if not all_text:
                return result_df
            
            # Preprocess text for topic modeling
            processed_docs = [self._preprocess_text(doc) for doc in all_text]
            processed_docs = [' '.join(doc) for doc in processed_docs if doc]
            
            if not processed_docs:
                return result_df
            
            # Fit CountVectorizer and LDA model if not fitted
            if not self.fitted:
                # Fit CountVectorizer
                self.count_vectorizer = CountVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                doc_term_matrix = self.count_vectorizer.fit_transform(processed_docs)
                
                # Fit LDA model
                self.lda_model = LatentDirichletAllocation(
                    n_components=5,  # 5 topics
                    random_state=42,
                    max_iter=10,
                    learning_method='online'
                )
                self.lda_model.fit(doc_term_matrix)
            
            # Transform each text column
            for col in text_columns:
                if col in df.columns:
                    # Preprocess text
                    processed_text = df[col].fillna('').astype(str).apply(self._preprocess_text)
                    processed_text = processed_text.apply(lambda x: ' '.join(x) if x else '')
                    
                    # Transform to document-term matrix
                    doc_term_matrix = self.count_vectorizer.transform(processed_text)
                    
                    # Get topic distributions
                    topic_distributions = self.lda_model.transform(doc_term_matrix)
                    
                    # Add topic features
                    for i in range(topic_distributions.shape[1]):
                        result_df[f'{col}_topic_{i}_prob'] = topic_distributions[:, i]
                    
                    # Get dominant topic
                    dominant_topics = np.argmax(topic_distributions, axis=1)
                    result_df[f'{col}_dominant_topic'] = dominant_topics
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting topic features: {str(e)}")
            return df
    
    def _extract_embedding_features(self, df):
        """
        Extract word embedding features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with embedding features
        """
        try:
            result_df = df.copy()
            
            # Text columns to process
            text_columns = ['description', 'notes']
            
            # Combine text from all columns
            all_text = []
            for col in text_columns:
                if col in df.columns:
                    all_text.extend(df[col].fillna('').astype(str).tolist())
            
            if not all_text:
                return result_df
            
            # Preprocess text for embeddings
            processed_docs = [self._preprocess_text(doc) for doc in all_text]
            processed_docs = [doc for doc in processed_docs if doc]
            
            if not processed_docs:
                return result_df
            
            # Check if Gemini API is available
            if is_api_available('gemini'):
                # Here you would implement Gemini API calls for embeddings
                # For now, we'll skip and use local embeddings
                logger.info("Gemini API available, but implementation pending. Using local embeddings.")
            
            # Check if OpenAI API is available
            if is_api_available('openai'):
                # Here you would implement OpenAI API calls for embeddings
                # For now, we'll skip and use local embeddings
                logger.info("OpenAI API available, but implementation pending. Using local embeddings.")
            
            # Fit Word2Vec model (local implementation)
            self.word2vec_model = Word2Vec(
                sentences=processed_docs,
                vector_size=100,  # 100-dimensional vectors
                window=5,
                min_count=1,
                workers=4,
                sg=1  # Skip-gram model
            )
            
            # Train Doc2Vec model (local implementation)
            tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_docs)]
            self.doc2vec_model = Doc2Vec(
                documents=tagged_docs,
                vector_size=100,  # 100-dimensional vectors
                window=5,
                min_count=1,
                workers=4,
                epochs=10
            )
            
            # Transform each text column
            for col in text_columns:
                if col in df.columns:
                    # Preprocess text
                    processed_text = df[col].fillna('').astype(str).apply(self._preprocess_text)
                    
                    # Calculate average Word2Vec vectors
                    word2vec_vectors = []
                    for doc in processed_text:
                        if doc:
                            # Get vectors for words in document
                            word_vectors = [self.word2vec_model.wv[word] for word in doc if word in self.word2vec_model.wv]
                            
                            if word_vectors:
                                # Average the vectors
                                avg_vector = np.mean(word_vectors, axis=0)
                                word2vec_vectors.append(avg_vector)
                            else:
                                # Use zero vector if no words found
                                word2vec_vectors.append(np.zeros(100))
                        else:
                            # Use zero vector for empty documents
                            word2vec_vectors.append(np.zeros(100))
                    
                    # Add Word2Vec features (first 10 dimensions)
                    word2vec_vectors = np.array(word2vec_vectors)
                    for i in range(min(10, word2vec_vectors.shape[1])):
                        result_df[f'{col}_word2vec_dim_{i}'] = word2vec_vectors[:, i]
                    
                    # Calculate Doc2Vec vectors
                    doc2vec_vectors = []
                    for doc in processed_text:
                        if doc:
                            # Infer vector for document
                            vector = self.doc2vec_model.infer_vector(doc)
                            doc2vec_vectors.append(vector)
                        else:
                            # Use zero vector for empty documents
                            doc2vec_vectors.append(np.zeros(100))
                    
                    # Add Doc2Vec features (first 10 dimensions)
                    doc2vec_vectors = np.array(doc2vec_vectors)
                    for i in range(min(10, doc2vec_vectors.shape[1])):
                        result_df[f'{col}_doc2vec_dim_{i}'] = doc2vec_vectors[:, i]
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting embedding features: {str(e)}")
            return df
    
    def _preprocess_text(self, text):
        """
        Preprocess text for NLP analysis
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed tokens
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove digits
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words
            tokens = [word for word in tokens if word not in self.stop_words]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
            # Remove short words
            tokens = [word for word in tokens if len(word) > 2]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return []
    
    def fit_transform(self, df):
        """
        Fit the feature extractor and transform the data
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        # Extract features
        result_df = self.extract_features(df)
        
        # Get feature columns
        feature_cols = [col for col in result_df.columns if col not in df.columns]
        
        if len(feature_cols) > 0:
            self.fitted = True
        
        return result_df
    
    def transform(self, df):
        """
        Transform new data using fitted feature extractor
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        if not self.fitted:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        # Extract features
        result_df = self.extract_features(df)
        
        return result_df