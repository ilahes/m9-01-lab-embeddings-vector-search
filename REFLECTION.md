# Reflection

## Query 1: "my laptop won't switch on"

The best match was `kb-02`, which explains how to power up a device that will not turn on.

The query and passage share only a few general words, such as "won't" and "on". The important expressions are different: the query uses "laptop" and "switch on", while the passage uses "device" and "power up". This shows that the embedding captured the similar meaning of these expressions instead of relying only on identical keywords.

## Query 2: "how do I stop being billed every month?"

The best match was `kb-05`, which explains how to cancel a subscription.

The query does not use the words "cancel" or "subscription". Instead, it describes the same intention using "stop being billed every month". The embedding understood that stopping recurring billing is semantically related to cancelling a subscription.

## Query 3: "access denied error when saving a file"

The best match was `kb-08`, which explains the access-denied error and how to obtain write permission.

This query has some direct word overlap with the passage, especially "access denied" and "error". It also contains related concepts such as saving a file and write permission. Both keyword overlap and semantic meaning helped this passage receive the highest score.

## Query 4: "where do I leave my car in the evening?"

The best match was `kb-01`, which explains where employees may park after 6pm.

The query does not use the words "park", "lot", or "after 6pm". The embedding still connected "leave my car" with parking and "in the evening" with the time after 6pm. This is a clear example of retrieval based on meaning rather than exact words.

## Overall observation

The results show that embeddings capture semantic relationships between different expressions. They can connect phrases such as:

* "switch on" with "power up"
* "stop being billed" with "cancel your subscription"
* "leave my car" with "park"
* "in the evening" with "after 6pm"

Therefore, embedding-based search can retrieve relevant passages even when the query and passage use different vocabulary.

## Optional stretch: Unknown query

For the query `"what's the wifi password?"`, the highest result was `kb-07` with a cosine similarity score of `0.6560`. However, that passage only explains how to reset an account password and does not contain the Wi-Fi password.

This demonstrates that vector search always returns the nearest available passages, even when the knowledge base does not actually contain an answer.

For this small dataset, a tentative similarity threshold of approximately `0.68` could be used. If the highest score is below the threshold, the system could respond that the knowledge base does not contain a sufficiently confident answer. In a real application, the threshold should be tested and calibrated using many known and unknown queries.
