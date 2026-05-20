from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class Document:
    doc: str = ""
    doc_id: int = 0
    num_chunks: int = 0       # doc-only chunks stored in the memory store
    context: str = ""         # shared context prepended during encoding (list[list[str]] format)
    n_context_chunks: int = 0 # leading chunks produced by the context prefix; stripped before storage

class ProtocolConstants:

    @staticmethod
    def expect(q: mp.Queue, constant):
        k, v = q.get()
        assert k == constant, f"expect {constant} but got {k}"
        return v
    
    @staticmethod
    def send(q: mp.Queue, constant, data=None, block=True):
        q.put((constant, data), block=block)
