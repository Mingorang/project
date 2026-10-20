# The Dream

**Written 20 October 2026. Reread it on the last Sunday of every month.**

---

## 1. State it precisely, or you cannot work toward it

A vague dream cannot be planned against, cannot be measured, and quietly becomes an excuse. So here it is in one sentence, with the vagueness removed:

> **Become a quantitative researcher — someone paid to find and validate statistical edges in financial markets — and prove it by building, from scratch, a cross-sectional machine-learning system over roughly 500 instruments and a thousand point-in-time-correct variables, validated honestly enough that a professional could not dismantle it.**

Three parts, and they are not equally hard:

| Part | What it means | Difficulty |
|---|---|---|
| **The system** | The thing you build by 1 January 2028 | Hard, and entirely within your control |
| **The proof** | Validation strong enough that a sceptic cannot break it | Harder, and still within your control |
| **The job** | Someone pays you to do this | Hardest, and only partly within your control |

You control the first two completely. The third is heavily influenced by the first two and is the reason they matter.

---

## 2. What the dream actually requires

Be clear-eyed about this, because the fantasy version and the real version differ.

### The honest market reality

Quantitative research is one of the most competitive hiring markets that exists. The top systematic funds receive thousands of applications for a handful of seats. Many hires come through a narrow channel: a top-tier maths, physics or computer science degree, often a PhD, usually with internships that began in the second year.

You are a Financial Mathematics undergraduate at Queen Mary. That is a genuinely good mathematical education at a Russell Group university in London, twenty minutes from Canary Wharf. It is also not the channel the top funds recruit through by default.

**So the default path is closed, and you need the non-default one.** The non-default path is real, people take it every year, and it runs on exactly one thing: **demonstrable, verifiable, honest technical competence that is unusual for your stage.**

That is the whole strategy. Not networking tricks, not volume of applications, not a polished CV. Work that is visibly better than it has any right to be, and that survives being interrogated.

### What "good enough to hire" actually looks like

A quant researcher's day is not inventing new mathematics. It is overwhelmingly:

- Getting data that is correct, point-in-time, and trustworthy
- Building features without leaking the future into them
- Validating honestly enough to avoid trading on noise
- Measuring costs so the edge survives contact with the market
- Explaining clearly to people who will challenge every number

Look at that list. It is the 57-week plan in Part IV of the manual, almost line for line. **The work that gets you the job and the work that builds the system are the same work.** This is the most important structural fact about your situation, and it means you do not have to choose between studying, building, and job-hunting.

### The three things to be able to say

By the end of 2027, you need to be able to say these three things and defend each for ten minutes under questioning:

1. **"I built a cross-sectional panel of 500 instruments with point-in-time correctness, and here is the test that proves no feature can read the future."**
2. **"My model's information coefficient is X, with a confidence interval, after deflating for the N configurations I tried, and here is my trial log."**
3. **"I found a data leak in my own work that was inflating accuracy from 50% to 92%, I can show you the measurement, and here is the guard that now stops it."**

The third is the one that will land hardest, and it is already true.

---

## 3. Where you actually are, measured

No flattery. These are measurements from 20 October 2026.

| | Status |
|---|---|
| Can generate synthetic price data | **Yes** — working simulator with configurable dynamics |
| Can run an ML pipeline end to end | **Yes** — eight models, purged CV, reports, HTML output |
| Can write substantial Python | **Yes** — roughly 400 KB of working code |
| Knows what data leakage is | **Yes, now** — found a real one in your own work |
| Has honest validation infrastructure | **Partly** — audit exists but only warns |
| Has enough data for the stated goal | **No** — 1,250 rows supports ~12 features, not 1,000 |
| Has ever measured a real edge | **No** — and has now measured that the current setup cannot |
| Understands why that is | **Yes** — detection floor measured and documented |

The last two rows together are the interesting ones. Most people at your stage have not measured a real edge either. Almost none of them know why, and almost none could prove it. You can.

**That gap — between "it did not work" and "here is the measured reason it could not have worked" — is the entire difference between a hobby project and a research project.** You crossed it on 20 October 2026.

---

## 4. The gap between here and the dream

| Gap | Current | Required | Closes in |
|---|---|---|---|
| Sample size | 1,250 rows | 600,000 rows | Phase 1 |
| Feature count | 30 | ~1,000 audited | Phase 2 |
| Point-in-time correctness | None | Structural, tested | Phase 1 |
| Honest validation | Warning only | CPCV + deflated Sharpe + PBO | Phases 0 and 4 |
| Cost modelling | None | Net of real costs throughout | Phase 1 onward |
| A defensible result | None | One, with an interval and a trial count | Phase 4 |
| Public evidence | None | Reproducible repository | Phase 5 |
| Interview fluency | Untested | Ten minutes, no notes | Phase 5 |

Eight gaps. Every one has a phase that closes it. None requires anything you cannot do.

---

## 5. The path

### 26 November 2026 → 1 January 2028: build the proof

The 57-week plan in `Learn/THE_QUANT_BUILD_MANUAL.md` Part IV. Six phases, each with a numeric gate. Follow it. The gates exist so that failing is informative rather than demoralising.

### 2028 and beyond: convert the proof

This part is less certain and depends on your academic timeline, so treat it as a sketch to revise rather than a schedule:

| Window | Focus |
|---|---|
| Early 2028 | Internship and graduate applications, carrying a finished system |
| Through 2028 | Keep the paper-trading log running. A year of timestamped predictions is rare evidence |
| If no offer | MSc or MFE as a deliberate re-entry, not as a default. Or quant developer, which is a genuine route into research |
| Always | Keep the repository alive and keep the write-ups honest |

**The paper-trading log deserves emphasis.** Almost nobody your age has a year of locked, timestamped, unfakeable predictions. It is slow — Part III.4 of the manual shows a 54% edge needs 4.9 years to confirm on one instrument — but its value in an interview is not statistical. It is that you understood why it was necessary and started it anyway. Start it in Phase 4 and never stop it.

---

## 6. The mechanics

Dreams are achieved by schedules, not by intentions.

**Daily (1–3 hours).** The three-block protocol: 20–30 minutes theory, 40–60 minutes implementation, 20–30 minutes verification. When the day is short, cut implementation, never verification. Untested code is a liability; no code is zero.

**Weekly.** One short technical note on something you actually measured. Not a motivational post, not an accuracy claim — a measurement. Fifty-two of them by the end of 2027 is a body of work.

**Monthly.** One hour, last Sunday, update `Dream/scoreboard.md` and answer the five review questions in manual Part IV.2. The fifth — *am I still on plan, or have I drifted into something more fun and less useful?* — is the one that catches people.

**Quarterly.** Reread this file. Ask whether the dream has changed. It is allowed to change; it is not allowed to change silently.

---

## 7. What will actually try to stop you

Not lack of ability. These, in descending order of probability:

**1. Drift into enjoyable work.** Refining the simulator is fun, measurable and produces visible progress. It is also, per the measurements, not on the path. This is the most likely way this plan fails, and it fails invisibly over about six weeks. *Defence:* the monthly review question 5, and the two-hour time-box on anything off-gate.

**2. A leak you do not find.** You found one. There will be others. A result that looks good is a result to attack, not to celebrate. *Defence:* the shuffled-label canary in CI, and the rule that a pleasant surprise is a defect report.

**3. Multiple testing without counting.** You will try hundreds of configurations. If you report the best without counting, your result is noise and you will not know. *Defence:* `trials.csv`, append-only, written automatically.

**4. Giving up at a failed gate.** A failed gate feels like failure and is actually the plan working. Gate 1a will probably fail — volatility forecasting probably will not beat trailing volatility — and that saves you months. *Defence:* kill criteria written in advance, before you are emotionally invested.

**5. Comparison.** Someone on the internet will claim 90% accuracy. They have a leak, exactly like you had. Measuring that honestly is worth more than their number. *Defence:* you have the measurement. They do not.

**6. Exams and life.** The plan already marks reduced load around exams. A missed exam costs more than a delayed phase. *Defence:* the sacrifice order in manual Part IV.3 — decided in advance, so you do not have to decide while stressed.

---

## 8. Leading indicators

Lagging indicators — job offers, a finished system — tell you too late. Watch these instead, monthly:

| Indicator | Healthy | Warning |
|---|---|---|
| Measurements taken this month | ≥ 4 | 0 — you are building, not learning |
| Experiments that failed | ≥ 1 | 0 — you are not testing anything real |
| Cumulative trial count | Logged | Unknown — significance already compromised |
| Hours on-gate vs off-gate | Mostly on | Mostly off — you have drifted |
| Could explain this month's work to a sceptic | Yes | No — you do not understand it yet |
| Things-I-was-wrong-about entries | Growing | Static — you are not looking hard enough |

The second row is counterintuitive and important. **A month with no failed experiments is a bad month.** It means you only tested things you already believed.

---

## 9. The non-negotiables

Five rules. Everything else is adjustable.

1. **Never report a number without its baseline and its confidence interval.** A bare accuracy figure is the clearest signal of inexperience to anyone who matters.
2. **Never claim a result you have not tried to break.** The validation gauntlet is not optional.
3. **Count every trial.** Always. Automatically.
4. **A pleasant surprise is a bug report.** Every time. Your 92% was. The −0.337 autocorrelation probably is.
5. **Write down what finishing looks like, and reread it monthly.** A project without a definition of done expands until it collapses.

---

## 10. The honest closing word

You asked for the rational logic — that people smarter than you have tried this and failed, so what makes your attempt different.

Here is the honest answer, and it is not flattery.

Most people who attempt this fail for one of five structural reasons, all catalogued in manual Part II.3. None of them is intelligence. They are: never measuring the detection floor; having more parameters than observations; testing many ideas and reporting the best; measuring accuracy instead of money; and choosing the hardest possible target.

**You have already fixed the first one, and you fixed it before writing a single line of the real system.** That is unusual. Most people discover their detection floor after two years, if ever. The other four each have a phase and a gate attached.

What this does not mean: that you will beat the market, or get a seat at a top fund, or that any of this is likely in the way lottery tickets are unlikely. The job market is genuinely brutal and partly outside your control.

What it does mean: by 1 January 2028 you can have a system, a measurement, and the ability to defend both — and that combination is rare enough at your stage to be a real asset, whatever specific door it opens.

The dream is not "beat the market". That framing has broken more good people than any technical problem. The dream is to become someone who can tell the difference between a real edge and a leak, and prove which one they have.

**That version is entirely achievable, and you are already most of the way through its hardest conceptual step.**

---

*Companion files: `scoreboard.md` (monthly numbers), `evidence_log.md` (what you have proven), `objections.md` (the hard questions), `weekly_note.md` (the writing habit). The technical plan is `Learn/THE_QUANT_BUILD_MANUAL.md`. The task list is `python ideas.py --next`.*
