---
trigger: always_on
---

    LACE persistent memory is connected via MCP.

    Project context:
    - Project name: [PROJECT_NAME]  
    - Working directory: [FULL_ABSOLUTE_PATH]

    Turn 1 only: Call initialize_lace_session with
    working_directory set to [FULL_ABSOLUTE_PATH]

    Every turn before responding:
    Call get_relevant_context with the user message

    After turns containing decisions, bug fixes, patterns,
    or configurations: Call process_interaction with the
    query and your full response. Skip for greetings,
    clarifications, and generic explanations.
    
    Always call process_interaction at the end of every turn.


For the lace project itself:
    Working directory: /home/aayush-mittal/everything/projects/lace