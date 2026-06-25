import logging

from llm_studio.src.utils.logging_utils import NoisyHttpRequestsFilter


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_noisy_http_requests_filter_removes_huggingface_metadata_requests():
    log_filter = NoisyHttpRequestsFilter()

    assert not log_filter.filter(
        _record(
            "HTTP Request: HEAD "
            "https://huggingface.co/MTSmash/EvaGPT-German-2B-new-tokenizer-2026-q1/"
            'resolve/main/tokenizer_config.json "HTTP/1.1 200 OK"'
        )
    )
    assert not log_filter.filter(
        _record(
            "HTTP Request: GET "
            "https://huggingface.co/api/models/MTSmash/"
            "EvaGPT-German-2B-new-tokenizer-2026-q1/tree/main?recursive=true"
            "&expand=false "
            '"HTTP/1.1 200 OK"'
        )
    )


def test_noisy_http_requests_filter_keeps_non_huggingface_http_requests():
    log_filter = NoisyHttpRequestsFilter()

    assert log_filter.filter(
        _record('HTTP Request: GET https://example.com "HTTP/1.1 200 OK"')
    )


def test_noisy_http_requests_filter_still_removes_patch_requests():
    log_filter = NoisyHttpRequestsFilter()

    assert not log_filter.filter(
        _record('HTTP Request: PATCH http://127.0.0.1:10101 "HTTP/1.1 200 OK"')
    )
