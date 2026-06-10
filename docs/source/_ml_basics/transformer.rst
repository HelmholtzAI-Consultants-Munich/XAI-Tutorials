Introduction to Transformers
=========================================

The Transformer architecture was introduced in the 2017 paper *Attention Is All You Need* by Vaswani et al. It was originally developed for natural language processing tasks, especially machine translation, although the architecture itself is generic and is now widely used in applications such as chatbots and vision systems. 

The example used in this tutorial is German to English translation, e.g., “Ich habe Hunger” to “I am hungry”.

For a short introduction to Transformers, click below:

<Video overview>

**Background (before transformers)**:

- Sequence models such as RNNs, LSTMs, and GRUs process data sequentially.
- This sequential processing makes training slow since it cannot be parallelized.


.. figure:: ../_figures/sequential-processing-1.png


- These models struggle with long range dependencies, where information from earlier tokens may be lost over time (especially when the input is long).


.. figure:: ../_figures/sequential-processing-2.png


**Key ideas:**

- Uses attention to model relationships between all tokens in the sequence.
- Includes self attention to relate tokens within the same sequence.
- Uses cross attention to connect input and output sequences.
- Applies masked self attention in the decoder to ensure predictions depend only on past tokens during training.


.. figure:: ../_figures/masked-self-attention.png


- Enables parallel computation, making training faster and more scalable.

**Architecture overview:**

- **Word embedding:** Converts tokens into vector representations, applied to both encoder and decoder inputs.
- **Positional encoding:** Adds word position information to embeddings, used in both encoder and decoder.
- **Encoder:** Processes the input into a context rich representation using stacked blocks with multi head self attention and feed forward layers.
- **Decoder:** Generates the output sequence using masked self attention and cross attention with the encoder output, based on the current and previously generated tokens.


.. figure:: ../_figures/arch-overview.png


- **Training vs testing:**
    - During training, the decoder uses the ground truth shifted to the right (therefore, starting with <start> token) and uses masked self-attention trick so that the self-attention only attends the current and previous tokens. This trick makes the training parallelizable utilizing the available ground truth.
    - During testing it generates output tokens step by step.
- **Number of blocks N:** The number of encoder and decoder blocks can be adjusted to make the model shallower or deeper. By default, N = 6.


.. figure:: ../_figures/encoder.png


**Variants of transformers:**

- **Modality-wise:**
    - Words (natural language processing tasks)
    - Images (using image patches, e.g. Vision Transformer)
    - Audio (using chunks of sound)
    - Biological sequences (e.g. DNA or protein sequences)


.. figure:: ../_figures/transformer-variants.png


- **Architecture-wise:**
    - **Encoder only models:** such as BERT
    - **Decoder only models:** such as GPT
    - **Encoder–decoder models:** the original Transformer design


Word embedding
=========================================


.. figure:: ../_figures/word-embedding.png


Neural networks, including Transformers, require vector inputs, so words must be converted into vectors. See this video for the introduction to the word embedding.

<Video Word Embedding>

**Motivation:** 

- A simple solution: one hot vectors are a simple solution, where each word is represented by a vector with one element set to one and the rest zero.
    - However, one hot encoding does not capture semantic similarity, since related and unrelated words have zero similarity.
- Therefore, we need word representations that encode meaning and relationships between words.


.. figure:: ../_figures/word-emb-ex-1.png


**Method:**

- Word embedding maps words into vectors that capture semantic similarity.
- In a good word embedding, related words are closer, and vector operations can reflect relationships between words.
    - e.g., queen - woman + man = king


    .. figure:: ../_figures/word-emb-ex-2.png


- In Transformers, text is first split into smaller chunks using byte pair encoding, then mapped to vectors through a learnable linear layer.
    - Possibly not be the best for some cases, e.g., hun-gry vs hun-ter
    - But it still improves data efficiency and generalization, e.g., walk, walk-ed, walk-ing.


    .. figure:: ../_figures/byte-pair-encoding.png


In this course we simplify the explanation by considering the whole words instead of split tokens as the word embedding.


Positional encoding
=========================================
