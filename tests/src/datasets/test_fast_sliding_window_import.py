def test_dataset_package_import_installs_fast_indexer():
    from llm_studio.src.datasets.fast_sliding_window import FastSlidingWindowDataset
    from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset

    assert CustomDataset is FastSlidingWindowDataset
