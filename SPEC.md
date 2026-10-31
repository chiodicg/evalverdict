# Spec: evalverdict (AI Eval Harness)

Status: **Approved; implementation in progress** · Date: 2026-10-31 · Source: `scope.md` (this spec is authoritative where they differ)

## Objective

A public, portfolio-grade Python tool that evaluates any LLM API or ML model against a test set and answers one question with evidence: **should we change this model?**

Point it at one or more targets, give it a test set and a config, and get metrics from nine families, each with a mandatory confidence interval, plus pass/fail against thresholds and a two-model comparison report.

**Users:** ML/AI engineers and assurance reviewers. The primary reader is someone assessing the author's evals and assurance skills from the repo, the README and a sample report.

**User stories**

- As an engineer weighing a model upgrade, I run both models on one test set and get a per-metric verdict based on the confidence interval of the *difference*.
- As an assurance reviewer, I can see the rubric, sample size, CI, seed, dataset hash and library versions behind every number.
- As a user with a classifier and an LLM on the same task, I get both in one report with one scorer contract.

**v1 includes all nine metric families, Inspect red teaming and Langfuse tracing.** The two-model generation comparison from `scope.md` is the first milestone, not the end of v1.

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | The harness owns the run. DeepEval, Inspect, Evidently and the rest are engines behind the scorer contract. | Reports can then mix families, and no single framework's data model leaks through. |
| D2 | The base class computes the CI, and `ScoreResult` cannot exist without one. | Makes "0.83 with no interval" unrepresentable rather than a convention. |
| D3 | Scorers return per-item values plus an aggregate function. Bootstrap re-aggregates on each resample. | F1, AUROC and similar are not means of item scores. The `scope.md` inputs (predictions, references, optional metadata) are unchanged. |
| D4 | Comparisons use a paired bootstrap: the same resampled items for both models, with the CI on the difference. | Two overlapping CIs do not show that models are equivalent, and pairing is the more sensitive test. |
| D5 | Thresholds gate on the CI by default, giving **pass**, **fail** or **inconclusive**. `gate: point` opts out per metric. | A point estimate that clears a threshold by less than its own noise should not be reported as a pass. |
| D6 | The judge is pinned to an exact model ID (aliases such as `~latest` are refused) and may not be a model under test. Override requires `allow_self_judging: true`, which is printed in the report. Demo judge: `anthropic/claude-sonnet-5-5`. Judge prompts never name the model being scored, and the report states that the judge shares a vendor with the baseline. | Otherwise score differences may reflect the judge rather than the models. Sonnet is allowed because it isn't under test, but it may favour Haiku's style, so that link must be visible to readers. |
| D7 | Primary metrics are declared in config. Only they drive the recommendation; other metrics are reported as secondary. | With dozens of metrics, some will differ by chance. |
| D8 | Python 3.12 with `pandas<3` pinned. | Evidently 0.7.23 fails on import under pandas 3.1 (see Feasibility). |
| D9 | LLM calls go through LiteLLM. Cost comes from the provider-reported figure, then a config price table, otherwise "not measured". A zero cost on non-zero token usage counts as not measured. `validate` refuses a paid run if any model has no cost source. | One route for Anthropic, OpenAI and OpenRouter. An estimated cost is never presented as a measured one, and the budget cannot be bypassed by a model whose cost reads as zero. |
| D10 | Failures are never dropped silently. Parse failures count as wrong and are reported as a rate. API errors are excluded from metrics and reported as a rate; above `max_error_rate` (default 5%) the run is marked invalid. Comparisons use items where both targets succeeded, and show that count. | A model should not improve its score by failing to answer. |
| D11 | Sampling parameters are sent only if every compared target supports them. Otherwise models run at provider defaults, and the manifest records every parameter actually sent, including reasoning settings. | LiteLLM's data marks Haiku 5.5 as not accepting sampling parameters and omits `seed`, so the demo cannot be pinned to equal settings or made deterministic. The report must say so. |
| D12 | Tracing is chosen in config: `backend: none` (default) or `langfuse`, with `mode: self_hosted` or `cloud`. Keys come from environment variables only. With `required: false`, an unreachable server gives a warning and a note in the report. With `required: true`, the run stops before any paid call. | The code path is the same in both modes, so the choice belongs to the user. A tracing outage should not waste spend or change scores. |
| D13 | Labels record their provenance, and the report names it. Claude-drafted, human-reviewed labels are reported as "judge vs reviewed labels", with the number changed in review, never as human labels. The report also states when labeller and judge share a vendor. | An AI-drafted label accepted by a reviewer is not independent human judgement, and review anchors on the draft. Claiming otherwise would be the kind of overstatement this tool exists to catch. |
| D14 | Every scorer and metric is configurable. The `scorers` list alone decides what runs, with nothing implicit. Each entry can set `enabled`, `name`, `metrics` (a subset of what the scorer offers), `params` and `bootstrap`. One type can appear several times under different names. Each scorer declares a typed parameter schema with defaults, so any value that affects a result is a parameter, not a constant. Unknown keys, unknown metrics and gates on disabled metrics fail `validate`. Custom scorers load by import path, and must subclass `Scorer`. | Users choose what to measure and how, and typos can't silently change a run. Subclassing keeps custom scorers under the mandatory CI. |
| D15 | Configurability has two limits. CIs can't be turned off: method, level and resample count are configurable, but not their presence. `compare` refuses a metric whose scorer settings differ between runs, and the report lists every non-default setting. | D2 must survive configuration, and scores computed with different settings aren't comparable. |

## Tech Stack

Versions below are the tested set; the lockfile pins exact versions.

| Purpose | Library |
|---|---|
| Runtime | Python 3.12, pydantic 2 (config and data models), Typer (CLI), Jinja2 (reports) |
| LLM access | litellm 1.106.x (Anthropic, OpenAI, OpenRouter) |
| LLM-as-judge, faithfulness, RAG trio | deepeval 4.3.x |
| Red teaming | inspect-ai 0.3.286, inspect-evals |
| Tracing | langfuse 5.1.x (OpenTelemetry-based) |
| Classification, regression, ranking helpers | scikit-learn 1.9.x |
| BERTScore, ROUGE | evaluate 0.4.6, bert-score, rouge-score |
| Fairness | fairlearn 0.15.x |
| Drift | evidently 0.7.23 |
| Explainability | shap 0.54.x, lime 0.2.0.1 |
| Quality | pytest, ruff, mypy |

Dependencies are installed through extras (`nlp`, `judge`, `redteam`, `tracing`, `drift`, `fairness`, `explain`, `all`) and imported lazily. The full install is about 2.2 GB because of torch, so `pip install evalverdict` alone must stay light.

## Commands

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]" -c requirements.lock

# Use
evalverdict validate -c configs/demo_generation.yaml   # no API calls; prints estimated call count
evalverdict run      -c configs/demo_generation.yaml   # runs targets, scores, writes runs/<id>/
evalverdict compare  runs/<baseline> runs/<candidate>  # refuses if dataset hashes differ
evalverdict report   runs/<id>                         # rebuilds report without any model calls
evalverdict tracing check -c configs/demo_generation.yaml  # tests the Langfuse connection, no model calls
evalverdict scorers list                               # all scorers and their metrics
evalverdict scorers describe bertscore                 # parameters and defaults for one scorer

# Self-hosted Langfuse (optional)
deploy/langfuse/setup.sh                       # generates secrets into a gitignored .env
docker compose -f deploy/langfuse/docker-compose.yml up -d    # UI at http://localhost:3000

# Quality
pytest                        # unit and integration tests, no network or spend
pytest -m live                # real API calls, spends money, never run by default
ruff check . && ruff format --check . && mypy src
```

**Exit codes:** 0 pass, 1 fail, 2 inconclusive, 3 invalid run or configuration error. With several targets, the worst result wins.

**Labelling (demo):** `evalverdict labels sample`, `labels export` and `labels import` run the blind workflow in the Metrics section.

## Project Structure

```
src/evalverdict/
  cli.py            CLI entry point
  config.py         Config schema
  data.py           TestCase, Prediction, dataset loading and hashing
  runner.py         Execution, concurrency, cache, budget, resume
  targets/          litellm, sklearn, http, python_callable
  scoring/          base.py (contract), bootstrap.py, compare.py, gates.py
  scorers/          classification, regression, generation, ranking, agentic,
                    robustness, fairness, operational, explainability
  perturb/          typos, case, paraphrase, out-of-distribution
  engines/          deepeval_judge, inspect_redteam, evidently_drift (third-party isolated here)
  tracing/          sink protocol, langfuse sink
  report/           templates and HTML/JSON rendering
rubrics/            Versioned judge rubrics (YAML)
labels/             Reviewed labels, provenance and review change log
deploy/langfuse/    Pinned self-hosting stack, bound to localhost
configs/            Demo configs, one per family plus the comparison
datasets/           Small, licence-checked test sets with a provenance README
examples/           Committed sample report
tests/              unit/, contract/, statistical/, integration/, live/, fakes/, fixtures/
docs/               Metric reference, limitations, configuration, telemetry
runs/               Output (gitignored)
```

## Interfaces

### Scorer contract

```python
@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    level: float        # e.g. 0.95
    method: str         # e.g. "percentile"
    n_resamples: int
    n_undefined: int    # resamples where the metric could not be computed


@dataclass(frozen=True)
class ScoreResult:
    name: str
    family: Family
    value: float
    ci: ConfidenceInterval      # required, no default
    n: int
    higher_is_better: bool
    per_item: tuple[ItemScore, ...]
    details: Mapping[str, Any]  # rubric, confusion matrix, reliability bins, ...

    def __str__(self) -> str:
        half = (self.ci.high - self.ci.low) / 2
        return f"{self.value:.2f} ±{half:.2f} on n={self.n}"


class Scorer(ABC):
    """Subclasses compute item values and an aggregate. The base class owns the CI."""

    class Params(BaseModel):  # every setting that affects a result, with a default
        model_config = ConfigDict(extra="forbid")

    metrics: ClassVar[tuple[str, ...]]  # the metric names this scorer can produce

    def score(self, predictions, references, metadata=None) -> ScoreResult: ...  # final

    @abstractmethod
    def item_values(self, predictions, references, metadata) -> ItemValues: ...

    def aggregate(self, values: ItemValues, idx: np.ndarray) -> float:
        return float(values.array[idx].mean())  # override for F1, AUROC, ...
```

**Bootstrap rules**

- Defaults: 2,000 resamples, 95% percentile interval, seed from config. All are configurable for a run or per scorer (D15). Reports show the exact interval as well as ±half-width, because intervals can be asymmetric.
- Perturbed variants of one source item are resampled as a cluster, so they do not inflate `n`.
- Undefined resamples (e.g. AUROC with one class present) are counted in `n_undefined`. Above 5%, the CI is flagged as unreliable.
- `n < 30` is flagged in the report.
- Matrices and diagrams are details, not scalars; their headline numbers (per-class F1, ECE) carry the CI.
- By default the CI covers test-item sampling only. With `repeats > 1` on non-deterministic models, repeats are clustered under their item. The report states which applies. Because the demo models can't be made deterministic (D11), the demo also runs a 30-item subset with `repeats: 3` and reports run-to-run variation.

For planning: at a proportion of 0.83 with n=120, a 95% CI is about **±0.07**, not ±0.04; ±0.04 needs about n=340. Lower-variance continuous metrics can be tighter. Demo sets should be sized with this in mind.

### Target contract

```python
class Target(Protocol):
    def predict(self, case: TestCase) -> Prediction: ...
```

`Prediction` carries output, optional class scores, optional tool trajectory, timings, usage, cost, retries and error. Target kinds are `litellm`, `sklearn`, `http` and `python`.

- TTFT and throughput come from streaming. TTFT means the first visible output token, not reasoning output. Latency is timed on the final attempt only, with retries counted separately.
- The report records concurrency, because it affects latency.
- `sklearn` models load through pickle/joblib, which can execute code. The docs must say to load only trusted files.

### Config

```yaml
name: haiku-vs-luna
seed: 42
budget_usd: 26.00          # £20 at 1.3399 USD/GBP; required for any paid run; the run stops when reached
dataset: datasets/summaries.jsonl
targets:
  baseline:  {kind: litellm, model: anthropic/claude-haiku-5-5}
  candidate: {kind: litellm, model: openai/gpt-6-luna}
judge:
  model: anthropic/claude-sonnet-5-5
tracing:
  backend: langfuse        # none (default) | langfuse
  mode: self_hosted        # self_hosted | cloud
  base_url: http://localhost:3000   # required for self_hosted; cloud defaults to https://cloud.langfuse.com
  required: false          # true stops the run before any paid call if Langfuse is unreachable
bootstrap: {n_resamples: 2000, level: 0.95, method: percentile}   # run default
scorers:
  - type: bertscore
    params: {model_type: roberta-large}
    metrics: [f1]                  # optional subset; default is all metrics
  - type: rouge
    enabled: false                 # stays in the config but doesn't run
  - type: judge
    name: summary_quality          # names let one type run several times
    params: {rubric: rubrics/summary_quality.yaml}
  - type: latency
    params: {percentiles: [50, 95]}
    bootstrap: {n_resamples: 5000} # per-scorer override
  - import: my_pkg.scorers:DomainScorer   # custom scorer; runs local code, trusted sources only
gates:
  summary_quality.score: {min: 0.75, primary: true}
  bertscore.f1:          {min: 0.70}
  latency.ttft_p95_s:    {max: 2.0}
```

`evalverdict scorers list` shows every scorer and its metrics, and `evalverdict scorers describe <type>` shows its parameters and defaults. Examples of configurable values: k for ranking, ECE bins, extraction tolerance, perturbation types and rates, fairness subgroup columns, drift methods and bins, judge model per scorer, and SHAP and LIME sampling caps.

### Run output

Each target gets `runs/<timestamp>-<name>/<target>/`, holding `manifest.json` (config, dataset and rubric hashes, seed, exact model IDs, library versions, git SHA if the project is a git repo), `predictions.jsonl`, `scores.json` and `report.html` (self-contained). A `run` with exactly two targets also writes the comparison to the run's top-level directory.

Model outputs and judge decisions are cached by content hash, so resuming or rebuilding a report costs nothing. Secrets never appear in any output.

## Metrics

| Family | v1 metrics | Engine |
|---|---|---|
| Classification | Accuracy, per-class precision/recall/F1, confusion matrix, AUROC, AUPRC, ECE, reliability diagram, LLM parse-failure rate | scikit-learn |
| Regression and extraction | MAE, RMSE, MAPE, R², exact match, tolerance match, schema validity | scikit-learn, pydantic |
| Generation | BERTScore, ROUGE, LLM-as-judge with visible rubric, judge-human agreement, faithfulness/hallucination, context precision, context recall, answer relevance | evaluate, deepeval |
| Ranking and retrieval | Precision@k, recall@k, MRR, nDCG | scikit-learn, harness code |
| Agentic | Task success, tool-selection accuracy against gold sequence, steps, loops, unsafe-action rate, cost per completed task | Harness code |
| Robustness | Score drop under typos, paraphrase, case changes, adversarial and out-of-distribution inputs | `perturb/`, Inspect for LLM adversarial suites |
| Fairness | Metrics by subgroup, demographic parity difference, equalised odds | fairlearn |
| Operational | p50/p95 latency, TTFT, throughput, cost per prediction, PSI and KL drift, prompt and output distribution shift | Harness code, evidently |
| Explainability | ML: SHAP, LIME. LLM: faithfulness of self-explanation to the answer | shap, lime, deepeval |

**Family-specific rules**

- **Judge:** rubrics are versioned YAML, and the report shows the full rubric text and its hash. Agreement with labels (weighted Cohen's kappa with CI) is reported whenever labels exist, under the provenance rule in D13. If they don't, the report says agreement is unmeasured.
- **Demo labelling order:** (1) with approval, generate target outputs and cache them. (2) Sample 50, 25 per model, shuffled with the model identity hidden, because the labeller is from the same vendor as the baseline. (3) Carla labels 15 of them independently, before seeing any draft. (4) Claude drafts rubric labels with reasons for all 50, before any judge scores exist, without seeing Carla's labels; drafts of the 15 are used only to measure draft quality. (5) Carla reviews the other 35, still blind, and changes are logged in `labels/`. (6) Only then does the judge score them.
- **Label reporting:** three results are shown separately, never pooled: judge vs Carla's 15 labels, judge vs the 35 reviewed labels, and Claude's drafts vs Carla's 15. With n=15, the human-agreement CI will be wide and falls under the `n < 30` flag. It shows whether the judge is broadly reliable, not a precise figure.
- **Robustness:** typo, case, paraphrase, adversarial text and OOD suites are harness code and run identically on classifiers and LLMs. Adversarial text means model-agnostic evasion edits: homoglyphs, zero-width characters and leetspeak. Paraphrases are LLM-generated, pinned and cached, so both targets see the same text. Inspect covers LLM adversarial suites only.
- **Drift:** a test-set run has no live traffic, so drift compares a reference file with a user-supplied current file. LLM shift uses numeric descriptors of prompts and outputs (length, refusal rate). Evidently's call order is `run(current, reference)`, and the wrapper must enforce this.
- **Explainability:** the README must say plainly that LLM self-explanation faithfulness is a weaker signal and is not equivalent to SHAP or LIME.
- **BERTScore:** `model_type` is pinned in config. Like any other scorer setting, a mismatch between runs is refused by `compare` (D15), because scores are not comparable across models.

## Feasibility

Tested 2026-10-31 on a real install (Python 3.12, macOS), offline and with no paid calls.

| Check | Result |
|---|---|
| ROUGE per-item via `evaluate` | Pass |
| BERTScore per-item via `evaluate`, using distilbert | Pass; identical scores to `bert_score` directly |
| DeepEval GEval with custom judge class, standalone `measure()` | Pass with a stub judge |
| Langfuse 5.1.0 without keys | Pass; warns and becomes a no-op |
| Inspect eval through mock model | Pass; per-sample scores are letters (`C`/`I`) that must be mapped to numbers |
| Inspect OpenRouter, Anthropic and OpenAI providers | Present |
| Fairlearn parity, equalised odds and `MetricFrame` | Pass |
| SHAP and LIME on a scikit-learn model | Pass |
| LiteLLM mock streaming and cost lookup | Pass for OpenAI and Anthropic routes; **OpenRouter route is unmapped** (`ModelNotMappedError`), handled by D9 |
| Evidently PSI and KL with pandas 3.1 | **Fail: import error** |
| Evidently PSI and KL with pandas 2.3.3 | Pass; the full check set also passes and `pip check` is clean |
| LiteLLM cost lookup for the demo routes | Haiku and Luna pass ($0.10/$0.50 per million tokens), Sonnet judge passes ($2/$10). The OpenRouter route tried for Gemini returned `0.0`, so D9 still guards that case |
| LiteLLM parameter data for the demo models | Haiku and Sonnet: `supports_sampling_params` is false. Luna: `seed` supported. Handled by D11; not yet confirmed against the live APIs |
| OpenRouter model list | `gpt-6-luna` listed. Its Claude IDs differ (`anthropic/claude-haiku-5.5`), so IDs are not portable across routes |

**Not yet tested, so the first tasks must cover these:** real API calls and TTFT; Langfuse ingestion with real keys, SDK 5.1 against the self-hosted server (latest 4.72.0), and Langfuse alongside Inspect; DeepEval's faithfulness and RAG metrics; `inspect-evals` suites (not installed in the test, and some may need Docker); `roberta-large` speed on CPU; whether metric loading works offline; OpenRouter cost on real streams.

## Risks

| Risk | Mitigation |
|---|---|
| Scope is large, and the design could be wrong across nine families | Build the contract and first milestone before any other family; add families only after it passes. |
| `evaluate` has had no release since 2025-09 and its README points LLM users to LightEval | Isolate behind one module. `bert_score` and `rouge-score` directly are a fallback; the BERTScore fallback gave identical scores. |
| Evidently depends on pandas internals, and its `.dict()` output encodes columns in strings | Pin both, and keep a contract test on the parsed output. |
| API spend across models, judges and perturbations | Required `budget_usd`, cache, call estimate in `validate`, offline tests by default. Judge calls will dominate: estimated at about $0.009 each against $0.0003 per target call (assuming 2,000 in/500 out and 1,500 in/300 out tokens), so the $26 budget buys roughly 2,900 judge calls. Faithfulness and RAG metrics make several calls per item, so demo set sizes must be planned against this and measured on a small sample first. |
| Sonnet as judge can't be pinned to a fixed temperature, so scores may vary between runs | Judge decisions are cached, and the 30-item repeat subset also measures judge variation. |
| Inspect calls models through its own clients, so LiteLLM's budget tracking doesn't see them | Each suite gets a sample limit and its own share of the budget. Its usage is read from the Inspect log and added to the run total. |
| The budget can overshoot slightly because calls in flight still complete | Stop issuing calls before the limit, allowing for concurrency, and report the final spend. |
| Self-hosted Langfuse runs six services. Langfuse recommends 4 cores and 16 GiB; this Mac has 16 GB, and Docker is allowed 3.53 GiB | The recommendation is not a tested minimum for a demo. Raise Docker's memory (8 GiB is a starting point), and test it alongside BERTScore. If it won't fit, run tracing separately from the BERTScore step or use `mode: cloud`. |
| Upstream Compose defaults include placeholder secrets and expose some ports | Ship `deploy/langfuse/` with a pinned image version, generated secrets in a gitignored `.env`, and all ports bound to `127.0.0.1`. |
| Red-team prompts are harmful by design and go through your own provider accounts | Check each provider's usage terms before the first adversarial run, keep suites small, and run them only with approval. |
| Third-party telemetry on a public tool | Disable it by default where supported and document what each library sends. DeepEval's opt-out flag is set in the test, but its effect is unconfirmed. |
| Latency differences reflect time of day, region or load | Report concurrency and timing window; treat cross-provider latency as indicative. |

## Testing Strategy

- **Framework:** pytest. Default tests use no network, keys or spend.
- **Known-answer tests:** every scorer against hand-computed or library-computed values.
- **Contract tests:** every registered scorer must return a valid `ScoreResult` with a CI, run with default params, reject unknown keys, and produce only the metrics selected. One parametrised test enforces this, so new scorers are covered automatically.
- **Statistical test:** on simulated data, 95% intervals must cover the true value 93–97% of the time, for a mean and for one non-mean metric. Paired comparison must show the expected false-positive rate on identical models.
- **Failure-path tests:** parse failures, API errors, budget stop, resume, single-class resamples, mismatched dataset hashes, zero-cost responses, unreachable Langfuse in both `required` settings, and `self_hosted` with no `base_url`.
- **Integration:** an end-to-end `run` then `compare` with LiteLLM `mock_response`, Inspect `mockllm` and a stub judge.
- **Live (`-m live`):** a small smoke test per provider, plus a Langfuse trace check. Run manually.
- **Coverage:** 85% line coverage on `scoring/` and `scorers/`; the other tests matter more than the number.

## Boundaries

**Always**
- Report every headline number with CI and n, through `ScoreResult`.
- Put any value that affects a result in the scorer's `Params`, with a default.
- Keep third-party imports inside `engines/` or the scorer that needs them.
- Run ruff, mypy and pytest before calling a task done.
- Save exact model IDs, seed, hashes and versions with each run.
- Test a new scorer against known answers before using it in a report.
- Update this spec first when a decision changes.

**Ask first**
- Any live run with real API spend, and any run above its stated budget.
- Adding a dependency beyond this spec, or changing a version pin.
- Changing the `ScoreResult` or `Scorer` contracts.
- Changing default gate, bootstrap or failure-handling behaviour.
- `git init`, commits, pushes, publishing to PyPI, or making the repo public. The repo goes on Carla's personal GitHub, never a work account, with a personal commit identity set before the first commit.
- Adding CI configuration.
- Adding any dataset to the repo, until its licence is confirmed.

**Never**
- Commit keys, `.env` files or personal data.
- Show a score without an interval.
- Present an unmeasured value as zero, or an estimate as measured.
- Weaken or delete a failing test or gate to get a pass.
- Run red-team suites against any system without its owner's authorisation.
- Claim LLM self-explanation is equivalent to SHAP or LIME.

## Success Criteria

**Milestone 1: the `scope.md` "done" line**

1. `evalverdict run` evaluates two different LLMs on one test set, producing BERTScore, judge scores with the rubric visible, TTFT, throughput and threshold results, all with CIs where a metric is numeric.
2. `evalverdict compare` produces a report with a paired-difference CI per metric, a verdict (better, worse, inconclusive) and a recommendation from primary metrics.
3. `compare` refuses runs with different dataset hashes.

**v1**

4. All nine families run from committed demo configs and appear in one mixed report, including a classifier and an LLM on the same classification task.
5. No code path can produce a headline number without a CI, shown by the contract test.
6. The coverage test (93–97%) and false-positive test pass.
7. Judge agreement is reported on the 50-item labelled subset as separate human-labelled (15) and reviewed (35) results, with provenance and the number changed in review, or the report states it is unmeasured.
8. A real Inspect adversarial suite runs against an LLM target, and results appear in the report under the scorer contract.
9. With `mode: self_hosted` against the local stack, a run's calls and scores appear as traces. Switching to `mode: cloud` needs only config and environment changes, with no code change, and is covered by a test. With `backend: none`, the same run completes with identical scores.
10. Rerunning from cache reproduces identical scores and intervals, with no model calls.
11. A paid run stops at `budget_usd`, keeps its partial results, and marks the run incomplete.
12. All default tests pass with no network, ruff and mypy are clean, and the coverage target is met.
13. The README has a quick start that works on a clean machine, a sample report, the limitations section and the explainability caveat.
14. Every scorer and metric can be enabled, disabled, selected and parameterised from config alone. A custom scorer works by import path. Invalid config fails `validate`, and no configuration can remove a CI.

## Resolved Questions

1. **Demo models:** `anthropic/claude-haiku-5-5` (baseline) vs `openai/gpt-6-luna` (candidate), both on direct provider routes so latency is comparable.
2. **Labels:** split between independent and reviewed labels; see 7 (D13).
3. **Langfuse:** selectable in config, with `self_hosted` and `cloud` modes. Carla will run it self-hosted locally (D12).
4. **Licence and name:** MIT. The name first proposed, `eval-harness`, is taken on PyPI by an active project with the same purpose (`regokan/evalh`), so the package and CLI are `evalverdict`, which was unclaimed on 2026-10-31.
5. **Judge:** `anthropic/claude-sonnet-5-5`, chosen by Carla over the third-vendor default first proposed.
6. **Demo budget:** £20, set as `budget_usd: 26.00` (1.3399 USD/GBP, 2026-10-30).
7. **Labelling:** Carla labels 15 items independently, and Claude drafts labels that Carla reviews for the other 35 (D13).

## Open Questions

None.
