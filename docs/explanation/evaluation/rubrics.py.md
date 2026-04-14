# evaluation/rubrics.py

## What this file does
Compositional rubric tree defining per-dimension and weighted scoring logic for representment episodes.

## Runtime role
- grading/evaluation module

## Key contents
- File size: 13781 bytes
- Approximate line count: 403
- Module docstring: OpenEnv Rubric subclasses that power ChargebackOps grading.

Every scoring dimension is a standalone :class:`openenv.core.rubrics.Rubric`
so the whole grader can be introspected via ``named_rubrics``, captured via
``state_dict``, and swapped piecewise (e.g. replace :class:`NoteQualityRubric`
with an ``LLMJudge``). The per-case composite uses :class:`WeightedSum` with
weights that must sum to 1.0.

The rubrics take their inputs via a :class:`GradingContext` dataclass passed
as the ``action`` argument of :meth:`Rubric.forward`. The ``observation``
argument is ignored — ChargebackOps grading operates over deterministic
episode progress, not on the last observation payload. This keeps the rubrics
pure and unit-testable without an environment instance.
- Top-level classes (11): GradingContext, EpisodeGradingContext, StrategyCorrectnessRubric, EvidenceQualityRubric, PacketValidityRubric, DeadlineComplianceRubric, EfficiencyRubric, OutcomeQualityRubric, NoteQualityRubric, CaseRubric, ChargebackOpsEpisodeRubric
- Top-level functions (4): _ratio, _final_resolution, _contest_is_valid, grade_representment_note

## Connections to other files
### Depends on / references
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- README.md
- docs/RESULTS.md
- docs/RUBRIC_AUDITOR_PRD.md
- evaluation/__init__.py
- evaluation/grading.py
- server/chargeback_ops_environment.py
- tests/test_grader.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
