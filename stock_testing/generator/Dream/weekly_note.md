# Weekly Technical Note

One per week. Fifty-two by the end of 2027 is a body of work that no CV line can match.

## The rule that makes this worth doing

**Post measurements and methods. Never post accuracy claims.**

"93% accuracy predicting the market" is read by every practitioner as *this person has a leak
and does not know it*. It is actively worse than posting nothing, because it tells them
something true about your experience level. You know this first-hand: yours was 92%, and it
was a leak.

What practitioners respect instead:

- A measurement with its baseline and interval
- A negative result, reported cleanly
- A method implemented and explained plainly
- A bug you found in your own work and what the tell was

## Template

```markdown
**[One concrete thing I measured this week]**

The setup: [two sentences — what, on what data, how validated]

The number: [the measurement, its baseline, its confidence interval]

What surprised me: [the honest bit — what you expected versus what happened]

What I am doing about it: [one sentence]

[chart if it genuinely helps; no chart if it does not]
```

## Worked example, from this project

> **My stock prediction model was 92% accurate. It was measuring nothing.**
>
> The setup: a Random Forest over 30 technical features, trained on simulated price paths from
> my own random-number generator. Test accuracy came back at 0.926 — on data with no signal in
> it by construction.
>
> The number: the identical model on randomly permuted labels scored 0.491. That gap is the
> whole story. One feature — the share of up-candles in the run — separated the label on its own
> at AUC 0.92, because the feature and the label were computed over the same finished path. The
> model was reading the answer, not forecasting it.
>
> What surprised me: my own code had been printing a leakage warning on every run for weeks. It
> warned and then trained anyway. I had read past it every time.
>
> What I am doing about it: the audit now raises instead of warning, and features come from the
> first 70% of each path with the label from the last 30%. Accuracy is now 0.50, which is the
> correct answer on random walks.

That post is more useful to a reader, and more impressive to a practitioner, than any accuracy
figure — because every claim in it is falsifiable and the author is the one who falsified it.

## Note log

| Week | Date | Topic | Posted |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
