# V14 balanced semantic repair

V14 is the final preregistered repair of the latent evaluation/deployment
scenario bank before the application deadline. It follows the completed V13
two-judge failure audit.

The repair is systematic rather than outcome-selective:

- every clean binary pair is rewritten with the same explicit role definitions;
- eval and deploy members have identical word multisets and differ only in the
  relation between roles and outcomes;
- qualitative model assessment is represented by a fifth extracted fact;
- internal operating staff are distinguished from external recipients;
- model-assessment outcomes are distinguished from live operational outcomes;
- the old final holdout is retired and replaced by 64 newly generated rows.

Both judges run through local Codex with ChatGPT subscription authentication:

- `gpt-5.6-luna`, reasoning effort `max`;
- `gpt-5.6-terra`, reasoning effort `high`.

`fast_mode` is explicitly disabled. User configuration is ignored, no priority
or fast service tier is requested, tools are disabled, each judgment has its own
ephemeral session, and completed rows are saved atomically. Production uses at
most 24 concurrent sessions, initially allocated 15 to Luna and 9 to Terra.

Both judges must independently pass every frozen semantic threshold. Consensus
cannot rescue a failure. A pass unlocks only bounded exact-FP8 source-feature
extraction. Any failure is a final semantic stop for this application cycle.
