# Eval harness for LLM endpoints and ML models

Point it at any LLM API or ML model, give it a test set, get BERTScore, LLM-as-judge scores with the rubric visible, latency (TTFT, throughput) and a pass/fail against thresholds you set in a config file. 
Done when it runs against two different models on the same test set and produces a comparison report, which is specially good for deciding whether upgrade/change models.

## Stack
Probably Python
I would like to use: 
- Deep eval for LLM as judge
- Inspect tool (AISI) for red teaming
- Langfuse for traces

Keep the interface identical: every scorer takes predictions, references and optional metadata, and returns a score plus per-item details, so a report can mix families. And make confidence intervals mandatory, bootstrap is enough; an eval that says “0.83” without “±0.04 on n=120” is the thing you’d flag in a client’s system, so the harness shouldn’t do it either.

Libraries to lean on rather than rewrite: scikit-learn for everything in the first two families and the fairness slices, evaluate from Hugging Face for ROUGE and BERTScore, Fairlearn if you want the fairness metrics done properly, Evidently for drift.


## Metrics

- Classification (ML classifiers, LLM labelling, routing, guardrail decisions). Accuracy, precision, recall, F1 per class, confusion matrix, AUROC and AUPRC where you have scores, calibration (expected calibration error, reliability diagram). For LLMs add a parse-failure rate: how often the output wasn’t a valid label at all. That one is often the real finding.
- Regression and extraction (numeric prediction, pulling values from documents). MAE, RMSE, MAPE, R². For LLM extraction add exact-match and tolerance-match rates, and schema validity when the output is structured.
- Generation (summaries, test cases, answers, your LLM bread and butter). BERTScore, ROUGE as a cheap baseline, LLM-as-judge with the rubric in the repo and judge agreement with human labels reported, faithfulness or hallucination rate against the source, and a RAG-specific trio where retrieval is involved: context precision, context recall, answer relevance.
- Ranking and retrieval (search, recommender, RAG retrievers). Precision@k, recall@k, MRR, nDCG. Same scorer serves a classical retriever and an embedding model.
- Agentic. Task success rate, tool-selection accuracy against a gold sequence, number of steps and loops, unsafe-action rate, cost per completed task.
- Then three cross-cutting families that apply to every task type, and which are where your assurance background shows:
- Robustness. Score drop under perturbation: typos, paraphrase, case changes, adversarial inputs, out-of-distribution samples. Same test runs on the classifier and the LLM.
- Fairness and bias. Performance sliced by subgroup, demographic parity difference, equalised odds. This is the metric family that assurance audits ask for and most eval tools ignore.
- Operational. Latency (p50, p95, TTFT and throughput for LLMs), cost per prediction, and drift: PSI or KL divergence between training and live input distributions for ML, prompt and output distribution shifts for LLMs.
- Explainability is the one place the two worlds don’t share a metric. LIME and SHAP for ML; for LLMs you can report faithfulness of a self-explanation to the answer, but it’s a weaker signal and worth saying so in the README rather than pretending they’re equivalent.


