---
description: Machine-readable trustcv documentation for AI coding assistants - llms.txt, llms-full.txt and api-schema.json, generated from the docs and the source on every release.
---

# For AI agents

Coding assistants write better trustcv code when they read the current API instead of guessing from older examples. The site publishes three machine-readable files, regenerated from these pages and from the source code on every build:

| File | Contents | Use it for |
| --- | --- | --- |
| [`llms.txt`](https://ki-smile.github.io/trustcv/llms.txt) | A short index of the documentation with links | Giving an assistant an overview |
| [`llms-full.txt`](https://ki-smile.github.io/trustcv/llms-full.txt) | The Get started, Guides and Reference pages as plain Markdown | Pasting full context into a chat or a project knowledge base |
| [`api-schema.json`](https://ki-smile.github.io/trustcv/api-schema.json) | Every documented class and function with its parameters, defaults and first docstring line, the package version and the overall status values | Tools that need exact signatures |

The schema is built from the same list of objects as the [API reference](api.md), so the two cannot disagree.

## How to use them

Most assistants do not fetch these files on their own. Point yours at them explicitly, for example:

<!-- notest: prompt text, not Python -->
```text
Read https://ki-smile.github.io/trustcv/llms-full.txt before writing code.
Use trustcv to cross-validate a model on patient data with repeated visits:
patient IDs go in groups=, preprocessing stays inside an sklearn Pipeline,
and report result.overall_status and the confidence intervals.
```

You can also install trustcv locally and let the assistant inspect it: `trustcv examples` lists seven complete recipes, and `help(trustcv.TrustCV)` shows the docstrings the reference is built from.

## What to check in generated code

Assistants still make the same mistakes people do. Before trusting generated code, check that:

- rows from the same patient are grouped (`groups=`) and a group-aware method is used;
- scaling, feature selection and resampling happen inside the `Pipeline`, not on the whole dataset;
- deep-learning early stopping uses an inner split of the training fold (`run(..., inner_validation=0.1)` on any trustcv runner), never the test fold;
- the code reads `result.overall_status` and treats `NEEDS_REVIEW` and `NOT_FULLY_VERIFIED` as open questions, not as a pass.

See [Reading results](../get-started/reading-results.md) for what each status means.
