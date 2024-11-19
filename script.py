# Import required modules and functions
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
from collections import Counter
import spacy

# Load the pre-trained English model for word vectors in spaCy
word2vec = spacy.load('en')

# Placeholder for a general category (used in entity finding)
blank_spot = "general topic"

# Responses the chatbot can give based on user input
response_a = "It seems like you're interested in {}. Can you tell me more about it?"
response_b = "Ah, {}! That's a fascinating topic. Do you have specific questions?"
response_c = "I'm happy to discuss {} with you. What would you like to know?"

# List of possible responses
responses = [response_a, response_b, response_c]

# ChatBot class implementation
class ChatBot:
    def find_intent_match(self, responses, user_message):
        """
        Finds the best response match for the user's message based on intent similarity.
        
        Args:
            responses (list): List of response templates.
            user_message (str): The message input from the user.
        
        Returns:
            str: The best-matching response template.
        """
        # Convert the user's message into a bag-of-words representation
        bow_user_message = Counter(preprocess(user_message))
        
        # Process all responses into bag-of-words representations
        processed_responses = [Counter(preprocess(response)) for response in responses]
        
        # Compute similarity for each response
        similarity_list = [compare_overlap(doc, bow_user_message) for doc in processed_responses]
        
        # Find the index of the response with the highest similarity score
        response_index = similarity_list.index(max(similarity_list))
        
        return responses[response_index]

    def find_entities(self, user_message):
        """
        Extracts entities (nouns) from the user's message and finds the most similar category.
        
        Args:
            user_message (str): The message input from the user.
        
        Returns:
            str: The most relevant entity extracted from the user's message.
        """
        # Perform POS tagging on the preprocessed user message
        tagged_user_message = pos_tag(preprocess(user_message))
        
        # Extract nouns from the tagged message
        message_nouns = extract_nouns(tagged_user_message)
        
        # Create spaCy tokens from the extracted nouns
        tokens = word2vec(" ".join(message_nouns))
        
        # Create a spaCy token for the blank spot category
        category = word2vec(blank_spot)
        
        # Compute similarity between nouns and the category
        word2vec_result = compute_similarity(tokens, category)
        
        # Sort results by similarity score in ascending order
        word2vec_result.sort(key=lambda x: x[2])
        
        # Return the noun with the highest similarity score
        return word2vec_result[-1][0]

    def respond(self, user_message):
        """
        Generates a response for the user's message.
        
        Args:
            user_message (str): The message input from the user.
        
        Returns:
            bool: Indicates that the response was generated.
        """
        # Find the best response based on intent match
        best_response = self.find_intent_match(responses, user_message)
        
        # Extract the most relevant entity from the user's message
        entity = self.find_entities(user_message)
        
        # Format the best response with the extracted entity
        print(best_response.format(entity))
        print("I hope I was able to help. See ya around!")
        
        return True

    def chat(self):
        """
        Starts the chatbot and listens for user input.
        """
        user_message = input("Hey! I'm a bot. Ask me your questions! ")
        self.respond(user_message)

# Instantiate the ChatBot class and start the chat
ChatBot = ChatBot()
ChatBot.chat()
