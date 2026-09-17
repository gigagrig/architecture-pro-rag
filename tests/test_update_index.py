"""Snapshot lifecycle tests; embeddings are deterministic and run offline."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np

import update_index
from rag_bot.retrieval import Retriever


class Encoder:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return np.array([[0, 1] if "NEW" in text else [1, 0] for text in texts], dtype=np.float32)


class UpdateTests(unittest.TestCase):
    def test_snapshot_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch('update_index.SentenceTransformer', Encoder), \
                patch('rag_bot.retrieval.SentenceTransformer', Encoder):
            root = Path(directory)
            knowledge = root / 'knowledge'
            knowledge.mkdir()
            index_dir = root / 'index'
            index_dir.mkdir()
            document = knowledge / 'entity_01.md'
            document.write_text('# Original\n' + 'old ' * 100)
            chunks, _ = update_index.load_chunks(knowledge, 100, 250, 40)
            update_index.write_chunks(index_dir / 'chunks.jsonl', chunks)
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1, 0]], dtype=np.float32))
            faiss.write_index(index, str(index_dir / 'faiss.index'))
            (index_dir / 'manifest.json').write_text(json.dumps(dict(model='test', dimension=2,
                chunk_count=1, chunk_min_words=100, chunk_max_words=250, chunk_overlap_words=40)))
            args = argparse.Namespace(knowledge_dir=knowledge, index_dir=index_dir, cache_folder=None,
                                      local_files_only=True, batch_size=16)
            retriever = Retriever(argparse.Namespace(index_dir=index_dir, cache_folder=None, local_files_only=True))
            self.assertEqual(update_index.update(args), 0)
            initial = update_index.active_directory(index_dir)
            self.assertEqual(update_index.update(args), 0)
            self.assertEqual(initial, update_index.active_directory(index_dir))
            added = knowledge / 'entity_02.md'
            added.write_text('# Added\n' + 'NEW ' * 100)
            self.assertEqual(update_index.update(args), 0)
            self.assertEqual(retriever.search('NEW', 1)[0]['title'], 'Added')
            self.assertEqual(retriever.index.ntotal, 2)
            document.write_text('# Changed\n' + 'changed ' * 100)
            self.assertEqual(update_index.update(args), 0)
            retriever.search('old', 2)
            self.assertNotIn('Original', [c['title'] for c in retriever.chunks])
            before_error = update_index.active_directory(index_dir)
            document.write_text('# Invalid\nshort')
            self.assertEqual(update_index.update(args), 1)
            self.assertEqual(before_error, update_index.active_directory(index_dir))
            document.unlink()
            self.assertEqual(update_index.update(args), 0)
            self.assertEqual(len(retriever.search('NEW', 5)), 1)
            self.assertEqual(retriever.index.ntotal, 1)
            # A broken published snapshot must not replace the loaded valid snapshot.
            (update_index.active_directory(index_dir) / 'manifest.json').write_text('{}')
            retriever.path = index_dir
            with self.assertLogs('rag_bot.retrieval', level='ERROR'):
                self.assertEqual(retriever.search('NEW', 1)[0]['title'], 'Added')


class RealUpdateTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('TASK6_REAL') == '1', 'Set TASK6_REAL=1 for cached E5 integration')
    def test_real_document_and_hot_reload(self):
        import shutil
        project = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix='task6-') as directory:
            root = Path(directory)
            shutil.copytree(project / 'knowledge_base', root / 'knowledge')
            index_dir = root / 'index'
            index_dir.mkdir()
            for name in ('faiss.index', 'chunks.jsonl', 'manifest.json'):
                shutil.copy(project / 'vector_index' / name, index_dir / name)
            args = argparse.Namespace(knowledge_dir=root / 'knowledge', index_dir=index_dir,
                cache_folder=project / '.cache/embeddings', local_files_only=True, batch_size=16)
            retriever = Retriever(argparse.Namespace(index_dir=index_dir,
                cache_folder=str(args.cache_folder), local_files_only=True))
            self.assertEqual(update_index.update(args), 0)
            initial_count = retriever.index.ntotal
            added = args.knowledge_dir / 'entity_39.md'
            shutil.copy(project / 'fixtures/task6_new_document.md', added)
            self.assertEqual(update_index.update(args), 0)
            results = retriever.search('Who coordinates the Lumen Relay Observatory?', 3)
            self.assertEqual(results[0]['title'], 'Lumen Relay Observatory')
            self.assertEqual(retriever.index.ntotal, initial_count + 1)
            update_index.emit('search_verified', title=results[0]['title'], score=results[0]['score'])
            current = update_index.active_directory(index_dir)
            self.assertEqual(update_index.update(args), 0)
            self.assertEqual(current, update_index.active_directory(index_dir))
            added.write_text(added.read_text().replace('Mira Solven', 'Taren Voss'))
            self.assertEqual(update_index.update(args), 0)
            self.assertIn('Taren Voss', retriever.search('Who coordinates the Lumen Relay Observatory?', 1)[0]['text'])
            added.unlink()
            self.assertEqual(update_index.update(args), 0)
            retriever.search('Who coordinates the Lumen Relay Observatory?', 3)
            self.assertEqual(retriever.index.ntotal, initial_count)
            update_index.emit('real_test_passed', checks=['add', 'search', 'unchanged', 'modify', 'delete', 'hot_reload'])
