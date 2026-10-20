# How an Antigravity-style automation works, in pseudocode, and how yours is wired

## What you are copying

Google Antigravity is an "agent-first" IDE. From the third-party guides I could find (the official docs at antigravity.google/docs/agent were only partly readable, so treat the details as unverified):

- An **Agent Manager** supervises several agent conversations at once.
- Agents report through **Artifacts** (task lists, implementation plans, screenshots, recordings, walkthroughs) instead of raw logs, and you can comment on an artifact to steer the run.
- Standing instructions live in **Rules** (and `AGENTS.md` / `GEMINI.md`), reusable procedures in **Workflows** and `SKILL.md` files, and tools such as a browser subagent and MCP servers.

I could not confirm how Antigravity itself triggers a workflow on a timer. The reliable pattern is: **the schedule comes from a scheduler, the work comes from an agent with rules and tools, and the result is an artifact a human reviews.**

## The pattern

```
DEFINE routine R:
    rules      = standing instructions                # your pasted prompt: focus on ML, 4 paths max, ...
    tools      = {shell, file read, market data, browser}
    guardrails = {read-only except reports/, never delete, caps on time and files, ask before risky actions}
    trigger    = schedule("08:30, Mon-Fri")  OR  manual("Run now")

WHEN trigger fires:
    context = rules + inputs_that_changed_since_last_run     # .toml / .txt / .md diffs
    plan    = agent.plan(goal = R.goal, context)             # Antigravity: implementation-plan artifact
    FOR each step in plan:
        result = agent.act(step, tools)                      # run a script, read files, fetch data
        IF NOT verify(result): retry once, else record the failure and continue
        IF step is risky: STOP and ask the human
    artifact = render(report)                                # HTML + Markdown brief
    append(metrics) to history                               # so tomorrow can compare with today
    notify(human)
HUMAN reviews artifact -> comments/edits rules -> next run is better
```

## Four ways to set it up for this project

| Path | Runs when | AI involved | Best for | Cost |
|---|---|---|---|---|
| **A. Claude scheduled task** (built for you: `ml-morning-routine`) | Mon–Fri, set for 08:30 (the app adds a few minutes of jitter and shows 08:42), while the Claude desktop app is open; a missed run happens on next launch | Yes: runs the script, reads the results, writes the brief and picks the next upgrade step | Same experience as a morning brief | Your Claude usage |
| **B. Windows Task Scheduler + the script** | Any time, app closed | No | Pure numbers, no summary | Free |
| **C. Your own `ml_model.py --daily-tip --daemon`** | While that terminal stays open, at `[daily_tip] time` in `config.toml` (currently `enabled = false`) | No | Tips built from your latest ML results; `python daily_ml_advisor.py --today` gives the rotating tip list | Free |
| **D. GitHub Actions cron** | In the cloud on a timer | No | Offloading work from your disk (your `prompt_ideas.txt` asks for free cloud options) | Free tier, check current limits |

Path B, one line in PowerShell (change the Python path if `python` is not on your PATH; this is not run for you):

```powershell
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "ML Morning Routine" /ST 08:30 /TR "python C:\Users\Admin\Documents\ml\project\stock_testing\generator\routine\daily_routine.py"
```

Path D caveat: a cloud runner cannot see your local `looped_analysis/` folder (7.6 GB, not in git), so only the market-snapshot half would work there.

## This routine, step by step (what `daily_routine.py` does)

```
1. digest_inputs():   hash config.toml, rf_tuning.toml, ideas.txt, prompt_ideas.txt, Learn/*  -> which changed?
2. audit_histories(): for every looped_analysis/ml_results/*/*/analysis_history.csv
        count rows with price <= 0 or no return
        train a forest on the "return" label, a forest on shuffled labels, and the base rate
        mechanical_gap = accuracy - max(base rate, shuffled accuracy)      # > 0.10 means leakage
3. market_snapshot(): for every instrument folder (stock_sap -> SAP, forex_gbp_usd -> GBPUSD=X) plus S&P 500, VIX, DXY
        returns, 20-day vol and its 5-year percentile, RSI, distance from the 200-day average
        scenario bands = daily sigma x 1 and x sqrt(5)                      # the "suggested movement"
4. honest_baseline(): expanding-window walk-forward on real daily bars, next-day direction
        lean = UP/DOWN only if accuracy beats chance band AND the majority baseline, else "none"
5. upgrade_progress(): is the code for each of the 5 upgrades present? (names listed in INSTITUTIONAL_UPGRADES.md)
6. write_report():    routine/reports/latest.html + latest.md + dated copy; one row appended to routine_log.csv
```

The "make the code learn and get closer to the new model" instruction from your prompt is implemented as a **measured gap**: each morning the log records the mechanical accuracy gap, the corruption rate, the honest edge and the number of upgrades done. The target ("new model") is mechanical gap near 0, corruption below 5%, honest verdict reported with a confidence interval, and 5/5 upgrades. The routine reads your `.toml` / `.txt` / `.md` files and tells you what changed; it never edits your code. You make the edits (that is what the five guides are for).

## Safety rails in this setup

- `daily_routine.py` is read-only against your project and writes only inside `routine/`.
- The scheduled task is told not to run `price_modeling.py` (interactive, and it writes into the 78,000-file folder), not to edit or delete anything, and not to push to git.
- Scenario bands are volatility ranges, not forecasts, and nothing here is investment advice.

## Change or stop it

- Time or days: edit the task in the Claude app's scheduled-tasks list (cron `30 8 * * 1-5`, local time).
- Run it right now: use "Run now" on the task, or `python routine/daily_routine.py` (add `--offline` to skip Yahoo Finance).
- Stop it: disable or delete `ml-morning-routine` in the same list.
