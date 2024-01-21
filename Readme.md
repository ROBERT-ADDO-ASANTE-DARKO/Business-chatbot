 # KYC Nations: The Ultimate Real Estate Bot

## Introduction

KYC Nations is a real estate chatbot that provides users with information about properties, neighborhoods, and the real estate market. The bot is powered by a language model that has been fine-tuned with instruction-tuning and RLHF, and it is able to provide accurate, factual, and thoughtful answers to user questions.

## How to Use the Bot

To use the bot, simply type in your question and the bot will respond with an answer. The bot is able to answer a wide variety of questions, including:

* What is the average price of a house in a certain neighborhood?
* What are the best schools in a certain area?
* What is the crime rate in a certain city?
* What are the pros and cons of living in a certain area?

## Code Overview

The code for the bot is written in Python and uses the following libraries:

* **Langchain:** A library for building language chains, which are sequences of language models that can be used to perform complex tasks.
* **Chainlit:** A library for building chatbots using Langchain.
* **Hugging Face Embeddings:** A library for loading and using pre-trained language embeddings.
* **FAISS:** A library for building and using vector stores.
* **CTransformers:** A library for loading and using pre-trained language models.

The code first loads the pre-trained language embeddings and vector store. Then, it creates a language chain that consists of a retrieval model and a language model. The retrieval model is used to retrieve relevant documents from the vector store, and the language model is used to generate answers to user questions.

Finally, the code creates a chatbot using Chainlit. The chatbot uses the language chain to answer user questions.

## Step-by-Step Explanation of the Code

The following is a step-by-step explanation of the code:

1. **Load the pre-trained language embeddings and vector store.**

```python
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
print(embeddings)
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
```

2. **Create a language chain that consists of a retrieval model and a language model.**