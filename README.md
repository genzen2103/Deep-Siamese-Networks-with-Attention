# Deep Siamese Network with Attention

Traditional Siamese Architecture can be improved significantly by incorporating Semantic information.

Here I have tried 3 ways to improve Accuracy for the task of Duplicate Question Detection in the Quora Dataset.

1. Using Glove Vector Embeddings

2. Using Bidirectional GRU to generate sentence embeddings

3. Attention mechanism based on Question context. ( Similar to Dynamic Coattention Networks )

All 3 models have been developed in Keras

Details:

Optimzer : Adam
Dataset : 1-million question-question labelled pairs from quora dataset
Epochs : 50
