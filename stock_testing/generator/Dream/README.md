# Dream/

The non-technical half of the project: what you are actually aiming at, how you will know
whether you are getting there, and the questions you need to be able to answer.

The technical half lives in `../Learn/THE_QUANT_BUILD_MANUAL.md`.

| File | What it is | When you open it |
|---|---|---|
| `dream.md` | The goal stated precisely, the honest market reality, the gap, the path, what will try to stop you | Monthly, and whenever you are drifting |
| `scoreboard.md` | The numbers that matter, with a monthly log | Last Sunday of every month |
| `evidence_log.md` | What you have **proven**, with the artefact for each claim | After every real result; before interviews |
| `objections.md` | The hard questions and your answers, marked solid / shaky / open | Whenever you make a non-obvious choice |
| `weekly_note.md` | Template for the weekly public technical note | Every week |

## The loop

```
daily     3 blocks: theory -> implementation -> verification
          python ideas.py --next        what to work on
weekly    one technical note on something you measured
monthly   scoreboard + the 5 review questions + reread dream.md
quarterly reread everything; ask whether the dream has changed
```

## Why this folder exists separately from Learn/

`Learn/` tells you *how* to build it. This folder tells you *whether you are still building the
right thing*, which is the question that actually decides the outcome.

The most likely way the next 14 months fail is not a technical error. It is six weeks of
enjoyable, measurable, off-gate work that feels like progress. That failure is invisible from
inside the code and obvious from inside `scoreboard.md`.

## The one rule

Nothing enters `evidence_log.md` without a reproducible artefact. If you cannot point at the
file or the command that proves it, it is not evidence — it is an intention.
