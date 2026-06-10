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


.. figure:: ../_figures/pos-enc.png


Positional encoding gives word position information to each word embedding. See this video below for an introduction to positional encoding.

<Video Positional Encoding>

**Motivation:**

- Transformers process all tokens in parallel rather than sequentially.


.. figure:: ../_figures/pos-enc-motivation-1.png


- The self attention mechanism is permutation equivariant, meaning that if we shuffle the input tokens, the outputs are shuffled in the same way.

    - However, the model does not know which word comes first or second as self-attention only compares the content of the tokens with each other, without any information about their positions.


    .. figure:: ../_figures/pos-enc-motivation-2.png


- However, word order changes meaning, for example when the positions of words in a sentence are swapped.

    - e.g., “dog bites man” vs “man bites dog”

- Therefore, Transformers need explicit position information to distinguish different token orders.

**Method:**

- Positional encoding is used to inject positional information to the model.It is a vector added to the word embedding, so it must have the same dimension as the embedding.


.. figure:: ../_figures/pos-enc-method-1.png

- A simple approach is binary position encoding, although it limits the maximum input length depending on the number of bits.

    - Max input length = {number of bits}^2. E.g., number of bits = 4 means max input length = 16


    .. figure:: ../_figures/pos-enc-method-2.png


- In the original Transformer, positional encoding uses sine and cosine functions.

    - Sine for even dimension ({2i}-th dimension) and cosine for odd dimension ({2i+1}-th dimension). Therefore, i=0 is for dimension 0 and 1, i=1 is for dimension 2 and 3, and so on if we have more dimension.
    - The equation also based on token/word position (pos) and vector dimension (d).
    - This approach does not impose a fixed maximum position.


    .. figure:: ../_figures/pos-enc-method-3.png


- Another common approach, especially in Vision Transformers, is to use learnable positional embeddings and let the model learn position information during training.


Self-attention
=========================================

Self attention mechanism is the core of transformer model. Self-attention is used in the encoder. In this subsection, we will discuss without the multi-head part.


.. figure:: ../_figures/self-att.png



Check the video below for introduction for more details:

<Video self attention>

**Motivation:**

- Context is essential for understanding word meaning. A word can have different meanings depending on surrounding words.

    - e.g., “The **bat** flies in the cave.” vs “He swings the **bat**.”
- Without considering other tokens in the sequence, the correct interpretation of a word cannot be determined.
- Self attention allows each word to look at other words in the sequence to gather contextual information.

**Intuition:**

- Each word tries to find which other words are important to it.
- Example #1, the word “bat”

    - In the sentence “the bat flies in the cave,” the word “bat” is connected to “flies” and “cave,” indicating it refers to an animal.
    - In the sentence “he swings the bat,” the word “bat” is connected to “swings” and “he” (a person), indicating it refers to a sports object.


.. figure:: ../_figures/self-att-intuition-1.png


- Example #2, the word “it”

    - In the sentence “I poured water into the cup until it was full,” “it” refers to “cup,” due to connection with something is “poured” “into” “until” “full”
    - In the sentence “I poured water from the bottle until it was empty,” “it” refers to “bottle,” due to connection with something is “poured” “from” “until” “empty”


.. figure:: ../_figures/self-att-intuition-2.png


- These connections between words help determine meaning, and self attention captures these relationships.

**Self Attention Mechanism:**

1. All word vectors are first combined into a matrix.
2. This matrix is multiplied by three learnable weight matrices to produce **query**, **key**, and **value** vectors.

    - Matrix multiplication is used instead of processing vectors one by one, allowing all tokens to be processed in parallel.


    .. figure:: ../_figures/self-att-mech-1.png


3. Each query is multiplied by the transposed keys to compute similarity scores using dot products.
4. The scores are normalized by the square root of the vector dimension, then passed through softmax to obtain attention weights.


    .. figure:: ../_figures/self-att-mech-2.png


5. These attention weights are multiplied with the value vectors and summed to produce updated word representations that incorporate context.


    .. figure:: ../_figures/self-att-mech-3.png


**Query, Key, Value Intuition:**

- **Query:** represents the word we are focusing on and asking what is relevant to it.
- **Key:** represents all words that compete for attention/relevancy with respect to the query.


.. figure:: ../_figures/self-att-qkv-1.png


.. figure:: ../_figures/self-att-qkv-2.png


- **Value:** contains the actual information of each word that will be aggregated, weighted by the attention score


.. figure:: ../_figures/self-att-qkv-3.png


Masked self-attention and cross-attention
=========================================

In the Transformer decoder, there are two types of attention mechanisms: masked self attention and cross attention. 

- Masked self-attention is the self-attention in the decoder block that masks future words to model relationships between current and past output tokens.
- Cross attention connects the decoder (output of masked self-attention) with the encoder output.


.. figure:: ../_figures/masked-self-att.png


