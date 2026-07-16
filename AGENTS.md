# LACE
## Decisions Made
* Update API documentation to reflect new `remember` parameter content as the primary argument, with `summary` and `body` noted as legacy fallbacks.
* Parallelize file reading in Vault Loading using `ThreadPoolExecutor` to cut down memory loading and graph building time at startup on multi-core systems.
* Use parameterized placeholder queries (`?`) for every SQLite query in relevant files.
* Refactor the config/CLI weights schema in `config.py` to match the actual multi-signal weights.
* Update the script to dynamically find the correct project scope, lengthen the test response, and check for the marker using a direct lookup (get()) instead of a vector similarity query.

## Patterns Established
* Run `lace generate-context` to synthesize your project's vault memories into a structured markdown file.
* The memory pipeline involves queuing interactions, running them through the worthiness gate, and extracting new markdown files if worthy.

## Debug Fixes and Known Issues
* Removed leftover prints from CLI user-prompting areas.
* Resolved issues with test harness assumptions, databases being uninitialized, project scope hardcoded to LACE but containing other projects, and test response too short and blocked by LACE's pre-filter.
* Identified unused dependency listed in `pyproject.toml` but never imported or used.

## Reference Information
* Uses `concurrent.futures.ThreadPoolExecutor` for parallel file parsing.
* No API keys, credentials, or secrets are hardcoded; they are loaded via env vars or the local `lace.yaml` file.
* Utilizes skills for security reviews, hardening configurations, and identifying common application flaws.

## Relevant Global Knowledge
* Doctor round-trip marker for LACE pipeline wiring.