# LACE Memory Protocol — Active

LACE persistent memory is connected via MCP. 

On EVERY turn without exception:
1. Call get_relevant_context FIRST with the user's message before thinking or responding.
2. Call process_interaction LAST after your complete response is generated.

This applies from turn 1. Do not wait to be reminded. 
Do not skip on simple messages. Do not skip on first turn.
These two calls bookend every single response you generate.
