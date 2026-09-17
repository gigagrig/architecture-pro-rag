#!/usr/bin/env python3
"""Evaluate RAG coverage with a fixed golden set and controlled corpus gaps."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

from rag_bot.cli import ScriptParser
from rag_bot.config import Settings
from rag_bot.llm import ChatGenerator
from rag_bot.query_log import QueryLog
from rag_bot.service import RagService


GAP_PATTERN = re.compile(r'\b(?:Xarn\s+Velgor|Velgor|Aeron(?:\s+Ardyn)?|Void\s+Cores?)\b', re.I)
REMOVED_DOCUMENTS = {'entity_02.md', 'entity_32.md'}


class CaptureLog(QueryLog):
    """Capture the very same event emitted by the production pipeline."""
    def __init__(self):
        self.event = {}

    def write(self, event: dict) -> None:
        self.event = event


def load_golden(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not 10 <= len(cases) <= 15 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError(f'{path}: expected 10–15 cases with unique IDs')
    for case in cases:
        for key in ('id', 'topic', 'question', 'expected_answer', 'keyword_groups', 'expected_sources', 'unknown_in'):
            if key not in case:
                raise ValueError(f'{path}: missing {key}')
        if not case['question'].strip() or any(not group for group in case['keyword_groups']):
            raise ValueError(f'{path}: empty question or keyword group')
    return cases


def score_case(case: dict, stage: str, event: dict) -> dict:
    expected_unknown = stage in case['unknown_in']
    keywords_ok = all(any(term.casefold() in event['answer'].casefold() for term in group)
                      for group in case['keyword_groups'])
    forbidden_ok = not any(term.casefold() in event['answer'].casefold()
                           for term in case.get('forbidden_answer_terms', []))
    allowed = set(case['expected_sources']) if not expected_unknown else set()
    sources = event['sources']
    matching = sum(Path(s['source']).name in allowed for s in sources)
    if expected_unknown:
        passed = event['status'] == 'unknown' and event['answer'].startswith('Я не знаю') and not sources
    else:
        passed = (event['successful_answer'] and event['answer_length'] >= 10 and keywords_ok
                  and bool(sources) and matching == len(sources) and forbidden_ok)
    return dict(expected_unknown=expected_unknown,
                expected_answer='Я не знаю' if expected_unknown else case['expected_answer'],
                passed=bool(passed and event['status'] != 'error'),
                keyword_groups_matched=keywords_ok if not expected_unknown else None,
                forbidden_answer_terms_absent=forbidden_ok,
                relevant_cited_sources=matching, cited_sources_count=len(sources),
                retrieval_hit=any(Path(s['source']).name in allowed for s in event['retrieved']) if allowed else None)


def prepare_experiment(source: Path, base_index: Path, root: Path, cache: Path, clarification: Path) -> tuple[dict, dict]:
    from update_index import active_directory, update
    active = active_directory(base_index)
    stages = {'baseline': active, 'restored': active}
    knowledge = root / 'gaps' / 'knowledge_base'
    knowledge.mkdir(parents=True)
    print(f'Created directory: {knowledge}', flush=True)
    audit = {'removed_entities': ['Xarn Velgor / Aeron Ardyn', 'Void Core'],
             'removed_documents': [], 'edited_documents': [], 'excluded_short_documents': [],
             'source_sha256': {}, 'gaps_sha256': {}}
    for path in sorted(source.glob('entity_*.md')):
        text = path.read_text(encoding='utf-8')
        audit['source_sha256'][path.name] = hashlib.sha256(text.encode()).hexdigest()
        if path.name in REMOVED_DOCUMENTS:
            audit['removed_documents'].append(path.name)
            continue
        title, body = text.split('\n', 1)
        sentences = re.split(r'(?<=[.!?])\s+', body.strip())
        kept = [s for s in sentences if not GAP_PATTERN.search(s)]
        if len(kept) != len(sentences):
            audit['edited_documents'].append(dict(source=path.name, removed_sentences=len(sentences)-len(kept)))
        cleaned = title + '\n\n' + ' '.join(kept) + '\n'
        if GAP_PATTERN.search(cleaned):
            raise ValueError(f'Gap entity remains in {path}')
        if len(re.findall(r'\S+', ' '.join(kept))) < 100:
            audit['excluded_short_documents'].append(path.name)
            continue
        target = knowledge / path.name
        target.write_text(cleaned, encoding='utf-8')
        print(f'Created document: {target}', flush=True)
        audit['gaps_sha256'][path.name] = hashlib.sha256(cleaned.encode()).hexdigest()
    gap_index = root / 'gaps' / 'vector_index'
    gap_index.mkdir()
    for name in ('faiss.index', 'chunks.jsonl', 'manifest.json'):
        shutil.copyfile(active / name, gap_index / name)
        print(f'Created initial index file: {gap_index / name}', flush=True)
    # Existing task-6 updater removes old chunks and embeds changed passages.
    args = argparse.Namespace(knowledge_dir=knowledge, index_dir=gap_index,
                              cache_folder=cache, local_files_only=True, batch_size=16)
    if update(args):
        raise ValueError(f'Cannot create gap index: {gap_index}')
    stages['gaps'] = gap_index
    # Restore documents from the verified original corpus, and publish a new version.
    restored = root / 'restored'
    shutil.copytree(knowledge, restored / 'knowledge_base')
    restored_index = restored / 'vector_index'
    restored_index.mkdir()
    for name in ('faiss.index', 'chunks.jsonl', 'manifest.json'):
        shutil.copyfile(active_directory(gap_index) / name, restored_index / name)
        print(f'Created restore seed: {restored_index / name}', flush=True)
    for path in sorted(source.glob('entity_*.md')):
        target = restored / 'knowledge_base' / path.name
        shutil.copyfile(path, target)
        print(f'Restored document: {target}', flush=True)
    target = restored / 'knowledge_base' / 'entity_17.md'
    shutil.copyfile(clarification, target)
    print(f'Replaced ambiguous article with focused fact card: {target}', flush=True)
    audit['clarified_document'] = dict(source=str(clarification), target=str(target),
        sha256=hashlib.sha256(clarification.read_bytes()).hexdigest())
    args.knowledge_dir, args.index_dir = restored / 'knowledge_base', restored_index
    if update(args):
        raise ValueError(f'Cannot create restored index: {restored_index}')
    stages['restored'] = restored_index
    audit['forbidden_mentions_in_gap_documents'] = 0
    return stages, audit


def summarize(rows: list[dict]) -> dict:
    known = [r for r in rows if not r['expected_unknown']]
    unknown = [r for r in rows if r['expected_unknown']]
    return dict(total=len(rows), passed=sum(r['passed'] for r in rows),
                expected_answers=len(known), correct_answers=sum(r['passed'] for r in known),
                expected_refusals=len(unknown), correct_refusals=sum(r['passed'] for r in unknown),
                false_refusals=sum(r['status'] == 'unknown' for r in known),
                unsupported_answers=sum(r['status'] == 'answered' for r in unknown),
                errors=sum(r['status'] == 'error' for r in rows),
                retrieval_hits=sum(r['retrieval_hit'] is True for r in known),
                failed_cases=[r['case_id'] for r in rows if not r['passed']])


def rescore(args: argparse.Namespace) -> int:
    """Reassess recorded answers after a reviewed golden-set correction."""
    print(f'Reassessment started: log={args.log}, run_id={args.run_id}, golden={args.golden}', flush=True)
    try:
        cases = {c['id']: c for c in load_golden(args.golden)}
        rows = [json.loads(line) for line in args.log.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows = [r for r in rows if r['run_id'] == args.run_id]
        if not rows:
            raise ValueError(f'No requests for run {args.run_id} in {args.log}')
        for stage in {r['stage'] for r in rows}:
            selected = [r for r in rows if r['stage'] == stage]
            if len(selected) != len(cases) or {r['case_id'] for r in selected} != set(cases):
                raise ValueError(f'Incomplete or duplicate golden cases for stage {stage}')
        results = []
        for row in rows:
            item = dict(row, original_passed=row['passed'])
            item.update(score_case(cases[row['case_id']], row['stage'], row))
            results.append(item)
        report = dict(run_id=args.run_id, method='Reassessment of recorded responses; no new LLM requests',
                      assessed_at=datetime.now(timezone.utc).isoformat(),
                      golden_sha256=hashlib.sha256(args.golden.read_bytes()).hexdigest(),
                      stages={stage: summarize([r for r in results if r['stage'] == stage])
                              for stage in dict.fromkeys(r['stage'] for r in results)},
                      changed_verdicts=[dict(stage=r['stage'], case_id=r['case_id'], answer=r['answer'],
                                            original_passed=r['original_passed'], passed=r['passed'])
                                        for r in results if r['original_passed'] != r['passed']])
        report['failures'] = sum(not r['passed'] for r in results)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Created assessment: {args.summary}; failures={report["failures"]}', flush=True)
        print('Reassessment finished', flush=True)
        return int(bool(report['failures']))
    except (OSError, ValueError, KeyError) as error:
        print(f'Reassessment failed for {args.log}: {error}', flush=True)
        return 1


def main() -> int:
    parser = ScriptParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python evaluate.py --experiment\n'
               'Single index: python evaluate.py --index-dir runtime/task7/RUN/gaps/vector_index --stage gaps\n'
               'Outputs: append-only JSONL requests, JSON summary, experiment corpus under work-dir.\n'
               'Exit: 0 all checks passed; 1 failed evaluation/setup; 2 invalid CLI.')
    parser.add_argument('--rescore', action='store_true', help='Grade existing --log without LLM calls; requires --run-id')
    parser.add_argument('--run-id', help='Run ID to select for reassessment')
    parser.add_argument('--experiment', action='store_true', help='Build gap/restored copies and evaluate all three phases')
    parser.add_argument('--knowledge-dir', type=Path, default=Path('knowledge_base'), help='Original corpus (default: knowledge_base)')
    parser.add_argument('--index-dir', type=Path, default=Path('vector_index'), help='Existing base index (default: vector_index)')
    parser.add_argument('--golden', type=Path, default=Path('golden_questions.txt'), help='Golden set in JSONL format')
    parser.add_argument('--cache-folder', type=Path, default=Path('.cache/embeddings'), help='Local E5 model cache')
    parser.add_argument('--clarification', type=Path, default=Path('fixtures/task7_quorin_clarification.md'),
                        help='Reviewed replacement for entity_17 in the restored experiment corpus')
    parser.add_argument('--stage', choices=['baseline', 'gaps', 'restored'], default='gaps', help='Expected-answer profile for a single index')
    parser.add_argument('--work-dir', type=Path, default=Path('runtime/task7'), help='Experiment copies directory')
    parser.add_argument('--log', type=Path, default=Path('logs.jsonl'), help='Append-only evaluation request log')
    parser.add_argument('--summary', type=Path, default=Path('tasks_artifacts/task7_summary.json'), help='Summary JSON (replaced each run)')
    args = parser.parse_args()
    if args.rescore:
        if not args.run_id or args.experiment:
            parser.error('--rescore requires --run-id and cannot be combined with --experiment')
        return rescore(args)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    report = dict(run_id=run_id, started_at=datetime.now(timezone.utc).isoformat(), stages={})
    failures = 0
    print(f'Evaluation started: {run_id}; index={args.index_dir.resolve()}; golden={args.golden.resolve()}', flush=True)
    try:
        cases = load_golden(args.golden)
        settings = replace(Settings.from_env(), cache_folder=str(args.cache_folder), local_files_only=True)
        report.update(model=settings.llm_model, min_score=settings.min_score, top_k=settings.top_k,
                      golden_sha256=hashlib.sha256(args.golden.read_bytes()).hexdigest())
        stages = {args.stage: args.index_dir}
        if args.experiment:
            stages, report['corpus_audit'] = prepare_experiment(args.knowledge_dir, args.index_dir,
                                                              args.work_dir / run_id, args.cache_folder, args.clarification)
        from rag_bot.retrieval import Retriever
        logger = QueryLog(args.log)
        print(f'Created or opened request log: {args.log.resolve()}', flush=True)
        for stage in ('baseline', 'gaps', 'restored'):
            if stage not in stages:
                continue
            settings = replace(settings, index_dir=stages[stage])
            retriever = Retriever(settings)
            if stage == 'gaps' and args.experiment and any(GAP_PATTERN.search(c['text']) for c in retriever.chunks):
                raise ValueError('Gap index still contains removed entity names')
            capture = CaptureLog()
            service = RagService(settings, retriever, ChatGenerator(settings), capture)
            rows = []
            for case in cases:
                print(f'{stage}: {case["id"]}: {case["question"]}', flush=True)
                try:
                    service.ask(case['question'])
                except Exception as error:
                    print(f'Request failed: {type(error).__name__}', flush=True)
                event = dict(capture.event, run_id=run_id, stage=stage, case_id=case['id'], topic=case['topic'],
                             golden_sha256=report['golden_sha256'])
                event.update(score_case(case, stage, event))
                logger.write(event)
                rows.append(event)
                print(f'{"PASS" if event["passed"] else "FAIL"}: {event["answer"]}', flush=True)
            report['stages'][stage] = dict(summarize(rows), index_dir=str(stages[stage]),
                                            documents=retriever.manifest['document_count'], chunks=retriever.index.ntotal,
                                            index_revision=str(retriever.path))
            failures += sum(not r['passed'] for r in rows)
    except Exception as error:
        # Setup errors never count as correct refusals.
        report['setup_error'] = f'{type(error).__name__}: {error}'
        print(f'Evaluation setup failed: {report["setup_error"]}', flush=True)
        failures += 1
    report.update(failures=failures, finished_at=datetime.now(timezone.utc).isoformat())
    try:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except OSError as error:
        print(f'Cannot write summary {args.summary}: {error}', flush=True)
        return 1
    print(f'Created summary: {args.summary}; appended log: {args.log}', flush=True)
    print(f'Evaluation finished; failures={failures}', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
