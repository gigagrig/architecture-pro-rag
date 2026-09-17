"""Offline tests for observable RAG outcomes and golden-set grading."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluate import load_golden, score_case, summarize
from rag_bot.llm import GenerationError
from rag_bot.query_log import QueryLog
from rag_bot.schemas import GeneratedAnswer
from rag_bot.service import RagService, ServiceBusy


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'queries.jsonl'
        self.chunk = dict(chunk_id='entity_01-chunk-001', title='Kael Ardyn',
                          source='knowledge_base/entity_01.md', score=0.9,
                          text='Kael is the twin brother of Lyra Voss.')
        self.retriever = Mock()
        self.retriever.path = Path('vector_index/revisions/test')
        self.retriever.search.return_value = [self.chunk]
        self.generator = Mock()
        self.generator.generate.return_value = GeneratedAnswer(
            answer='Сестра Kael Ardyn — Lyra Voss.', explanation=['В документе названа сестра.'],
            unknown=False, evidence=[dict(chunk_id=self.chunk['chunk_id'], quote=self.chunk['text'])])
        self.service = RagService(SimpleNamespace(top_k=5, min_score=0.7, llm_model='test', index_dir=Path('test')),
                                  self.retriever, self.generator, QueryLog(self.path))

    def event(self):
        return json.loads(self.path.read_text().splitlines()[-1])

    def test_successful_answer_logs_actual_retrieval_and_sources(self):
        self.service.ask('Кто сестра?')
        row = self.event()
        self.assertEqual(row['status'], 'answered')
        self.assertTrue(row['successful_answer'])
        self.assertEqual(row['answer_length'], len(row['answer']))
        self.assertEqual(row['retrieved'][0]['chunk_id'], self.chunk['chunk_id'])
        self.assertEqual(row['sources'][0]['chunk_id'], self.chunk['chunk_id'])
        self.assertIn('timestamp', row)
        self.assertEqual(row['index_revision'], 'vector_index/revisions/test')

    def test_no_chunks_is_a_refusal_not_a_success(self):
        self.retriever.search.return_value = []
        self.service.ask('Нет данных')
        row = self.event()
        self.assertFalse(row['has_retrieved_chunks'])
        self.assertFalse(row['successful_answer'])
        self.assertEqual(row['status'], 'unknown')
        self.generator.generate.assert_not_called()

    def test_filtered_search_results_are_distinct_from_eligible(self):
        self.retriever.search.return_value = [dict(self.chunk, score=0.1)]
        self.service.ask('Нерелевантный вопрос')
        self.assertTrue(self.event()['has_retrieved_chunks'])
        self.assertEqual(self.event()['eligible'], [])

    def test_generation_error_is_not_unknown_and_has_no_secret(self):
        self.generator.generate.side_effect = GenerationError('secret-key-for-test')
        with self.assertRaises(GenerationError):
            self.service.ask('Ошибка')
        row = self.event()
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['error_type'], 'GenerationError')
        self.assertNotIn('secret-key-for-test', self.path.read_text())
        self.assertFalse(row['successful_answer'])
        self.assertTrue(self.service.slot.acquire(blocking=False))
        self.service.slot.release()

    def test_busy_request_is_logged_without_releasing_occupied_slot(self):
        self.service.slot.acquire()
        with self.assertRaises(ServiceBusy):
            self.service.ask('Занято')
        self.assertEqual(self.event()['error_type'], 'ServiceBusy')
        self.assertFalse(self.service.slot.acquire(blocking=False))
        self.service.slot.release()

    def test_invalid_evidence_is_logged(self):
        self.generator.generate.return_value.evidence[0].chunk_id = 'invented'
        self.service.ask('Некорректная ссылка')
        self.assertEqual(self.event()['reason'], 'invalid_evidence')
        self.assertEqual(self.event()['status'], 'unknown')

    def test_log_failure_preserves_answer_and_releases_slot(self):
        with patch.object(QueryLog, 'write', side_effect=OSError('disk unavailable')):
            with self.assertLogs('rag_bot.query_log', level='ERROR'):
                self.assertFalse(self.service.ask('Кто сестра?').unknown)
        self.assertTrue(self.service.slot.acquire(blocking=False))
        self.service.slot.release()

    def test_grading_catches_wrong_fact_and_wrong_source(self):
        self.service.ask('Кто сестра?')
        row = self.event()
        case = load_golden(Path('golden_questions.txt'))[0]
        self.assertTrue(score_case(case, 'gaps', row)['passed'])
        self.assertFalse(score_case(case, 'gaps', dict(row, answer='Сестра — другой персонаж.'))['passed'])
        self.assertFalse(score_case(case, 'gaps', dict(row, sources=[dict(source='other.md')]))['passed'])

    def test_grading_distinguishes_correct_refusal_from_error(self):
        self.retriever.search.return_value = []
        self.service.ask('Нет данных')
        case = load_golden(Path('golden_questions.txt'))[-1]
        row = self.event()
        self.assertTrue(score_case(case, 'gaps', row)['passed'])
        graded = dict(row, **score_case(case, 'gaps', dict(row, status='error')), case_id=case['id'])
        graded['status'] = 'error'
        self.assertFalse(graded['passed'])
        self.assertEqual(summarize([graded])['errors'], 1)
        self.assertEqual(summarize([graded])['correct_refusals'], 0)

    def test_golden_profiles_have_required_coverage(self):
        cases = load_golden(Path('golden_questions.txt'))
        self.assertEqual(sum('gaps' not in c['unknown_in'] for c in cases), 7)
        self.assertEqual(sum('gaps' in c['unknown_in'] for c in cases), 4)
        gap = next(c for c in cases if c['id'] == 'gap_01')
        self.assertNotIn('restored', gap['unknown_in'])

    def test_extra_wrong_apprentice_fails_refined_grader(self):
        self.service.ask('Кто сестра?')
        row = dict(self.event(), answer='Учениками являются Orin Valis и Aeron Ardyn.',
                   sources=[dict(source='entity_17.md')])
        case = next(c for c in load_golden(Path('golden_questions.txt')) if c['id']=='known_06')
        self.assertFalse(score_case(case, 'restored', row)['passed'])
        self.assertTrue(score_case(case, 'restored', dict(row, answer='Ученик — Orin Valis.'))['passed'])

    def test_verified_secondary_weapon_source_is_allowed(self):
        self.service.ask('Кто сестра?')
        row = dict(self.event(), answer='Arcblade — лазерный меч.',
                   sources=[dict(source='entity_36.md'), dict(source='entity_28.md')])
        case = next(c for c in load_golden(Path('golden_questions.txt')) if c['id']=='known_03')
        self.assertTrue(score_case(case, 'restored', row)['passed'])

    def test_offline_reassessment_and_incomplete_log(self):
        from contextlib import redirect_stdout
        import io
        from evaluate import rescore
        cases = load_golden(Path('golden_questions.txt'))
        rows = []
        for case in cases:
            unknown = 'gaps' in case['unknown_in']
            answer = 'Я не знаю' if unknown else 'Ответ: ' + case['expected_answer']
            rows.append(dict(run_id='test-run', stage='gaps', case_id=case['id'], passed=False,
                             status='unknown' if unknown else 'answered', answer=answer,
                             answer_length=len(answer), successful_answer=not unknown,
                             sources=[] if unknown else [dict(source=case['expected_sources'][0])], retrieved=[]))
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary = Path(self.temp.name) / 'summary.json'
        args = SimpleNamespace(log=self.path, run_id='test-run', golden=Path('golden_questions.txt'), summary=summary)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rescore(args), 0)
            self.assertEqual(json.loads(summary.read_text())['failures'], 0)
            self.path.write_text(json.dumps(rows[0]) + '\n')
            self.assertEqual(rescore(args), 1)
