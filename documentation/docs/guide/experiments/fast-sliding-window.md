# Fast Sliding Window startup

Sliding Window keeps all tokens from long training samples, but it needs an exact
index of the token windows before training begins. LLM Studio builds this index
with batched tokenizer calls and stores it in a content-addressed cache.

The first start is faster because only token lengths are computed and no full
training tensors are materialized. Later starts with the same dataset, tokenizer,
maximum length, overlap, and formatting configuration reuse the cached index and
start close to the speed of Truncate.

By default, the cache is stored next to the training dataset in
`.h2o_llmstudio_cache`, or in `~/.cache/h2o_llmstudio` when the dataset path is
not available. Set `H2O_LLM_STUDIO_CACHE_DIR` to choose another cache directory.
Set `H2O_LLM_STUDIO_DISABLE_SAMPLE_INDEX_CACHE=1` to disable it.

The cache key includes the dataset contents and all settings that affect token
lengths or window boundaries. Changing the dataset, tokenizer, maximum length,
overlap, chat formatting, or raw Text handling automatically creates a new index.
