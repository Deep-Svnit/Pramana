from powermind_rag.config import RAGConfig


def test_default_device_is_gpu():
    assert RAGConfig().device == "cuda"
    assert RAGConfig().cuda_arch == "sm_120"
