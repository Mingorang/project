# Maths roadmap: from A-level Further Maths to the geometry of a market crash

*Where you are in mathematics, geometry and statistics, measured against what "The Geometry of a Market Crash" and this
project use, and a staged path to close the gap. Written 22 October 2026. Companion to
`GEOMETRY_OF_A_MARKET_CRASH.md`; fits inside Part IV of `THE_QUANT_BUILD_MANUAL.md`.*

---

## 1. Where you are now

What you told me: A-level Mathematics and Further Mathematics (Edexcel), with the options **Further Pure 1** and
**Further Mechanics 1**, and the wedge product did not feel difficult. What your manual adds: you are at QMUL, with
Financial Mathematics (MTH6154/MTH6155), Time Series (MTH6139), Machine Learning and stochastic-calculus material mapped
in Part XI. I do not know
which QMUL modules you have already completed, so the right-hand column is yours to tick.

### Geometry and linear algebra

| Topic | Where you met it | Status | Tick when done at QMUL |
|---|---|---|---|
| Vectors in 2D/3D, scalar product, angles, lines and planes | Core Pure | **solid** | |
| Vector (cross) product, area of a parallelogram | FP1 | **solid** | |
| Scalar triple product, volume of a parallelepiped | FP1 | **solid**: this *is* the 3D wedge product | |
| Matrices, determinant as area/volume scale factor and orientation, inverses, linear transformations (2×2, 3×3) | Core Pure | **solid** | |
| Complex numbers, $e^{i\theta}$, rotations | Core Pure | **solid**: the 2D geometric product behaves the same way | |
| Vectors in $\mathbb{R}^n$ for large $n$ (a 60-day return series is a point in $\mathbb{R}^{60}$) | — | **new** | |
| Span, linear independence, basis, dimension, rank | — | **new** | |
| Inner products, orthogonality, projections, Gram–Schmidt, QR | — | **new** | |
| Determinants of $n \times n$ matrices as $n$-dimensional volume | — | **new**, a short step from Core Pure | |
| Eigenvalues and eigenvectors, diagonalisation | Edexcel FP2 (not taken) | **new** | |
| Symmetric matrices, spectral theorem, positive definiteness, Cholesky | — | **new** | |
| Singular value decomposition, principal component analysis | — | **new** | |
| Wedge product, $k$-vectors, geometric product | — | **new**, closest to FP1 | |

### Statistics and probability

| Topic | Where you met it | Status | Tick when done at QMUL |
|---|---|---|---|
| Probability, conditional probability, independence, tree diagrams (Bayes) | A-level Maths | **solid** | |
| Binomial and normal distributions, hypothesis tests | A-level Maths | **solid** | |
| PMCC (calculator value) and its significance test for $\rho = 0$; interpreting a given regression line | A-level Maths (the formula for $r$ and least-squares regression are Further Statistics 2, not taken) | **solid** for interpretation and the test; the formulas are **new** | |
| Geometric series and sum to infinity | A-level Maths (Pure) | **solid** | |
| Geometric distribution, expected value $E(X) = 1/p$ (expected regime length) | Edexcel Further Statistics 1 (not taken) | **new**, a short step from geometric series | |
| Random vectors, mean vector, covariance matrix, correlation matrix | — | **new** | |
| Multivariate normal distribution | — | **new** | |
| Multiple regression as a projection, $R^2$, partial correlation, VIF | — | **new** | |
| Sampling error of correlation matrices, shrinkage (Ledoit–Wolf) | — | **new** | |
| Markov chains, transition matrices, stationary distributions | — | **new** | |
| Hidden Markov models: forward algorithm, Viterbi, EM | — | **new** | |
| Stationarity, autocorrelation, ARMA, GARCH | QMUL Time Series (MTH6139) | depends on QMUL | |
| Monte Carlo null distributions, chance levels | this project | **practised**: the detection floor, the backtest's chance level | |

### What the project has already taught you

You have used, in code, several ideas that the courses above formalise: rolling windows (local stationarity), purged
cross-validation (dependence in time), the detection floor (sampling error), Monte Carlo chance levels, and the Cholesky
factor for correlated paths (manual VI.7). The roadmap below turns those from techniques into understood mathematics.

### The honest summary

You are strong at **3D geometry and determinants**, which is the hardest *intuition* in the video. What is missing is
the **$n$-dimensional language** (independence, rank, inner products, eigenvalues) and the **multivariate statistics**
that tie that language to data. The video's idea sits where three chains meet:

```text
GEOMETRY      vectors in R^n -> independence and rank -> inner products, Gram–Schmidt -> n-D determinants
                                                                                       -> Gram determinant -> WEDGE VOLUME
STATISTICS    covariance matrix -> correlation matrix -> eigenvalues / PCA, regression R^2 / VIF  ----^
PROBABILITY   conditional probability -> Markov chains -> hidden Markov models (forward algorithm) -> REGIMES
```

None of it is far from you. Each stage below starts from something you already do well.

---

## 2. The path: five stages

Time estimates assume the theory block of your weekly timetable (`WEEKLY_TIMETABLE.md`) and the manual's three-block
protocol: **theory → implementation → verification**. Every stage ends with a project exercise, because the manual's
rule holds here too: if you cannot point at the code that proves it, you have not learned it yet.

### Stage 0 · Re-read what you know as geometry · 1–2 weeks · now

**Learn.** Almost nothing new: re-derive what you know in the language of the video. The one new piece is the PMCC
formula in the last bullet.

- Lagrange's identity, $|a \times b|^2 + (a\cdot b)^2 = |a|^2|b|^2$, from FP1 results; then its $n$-dimensional form
  by expanding both sides (geometry notes, Exercise 1).
- The scalar triple product as $\det[a\;b\;c]$, and why swapping two vectors changes its sign (orientation).
- The PMCC as the cosine of the angle between two *centred* data vectors. A-level Maths gave you $r$ from the
  calculator (9MA0 does not require the formula). It is $S_{xy}/\sqrt{S_{xx}S_{yy}}$ written with vectors:
  $r = \dfrac{(x-\bar x)\cdot(y-\bar y)}{|x-\bar x|\,|y-\bar y|}$.

**Project exercise.** Take **three** consecutive daily returns for two sectors: two vectors in $\mathbb{R}^3$. Compute the
area of their parallelogram with the cross product (FP1), then with $\sqrt{\det G}$. Same number. You have just computed
a 2-blade in exactly the dimension where you already know the answer.

**Done when** you can explain, without notes, why "the volume of nine unit vectors" is a correlation measure.

### Stage 1 · Linear algebra in $\mathbb{R}^n$ · 6–8 weeks · November 2026 to mid-January 2027

**Learn.** Vectors in $\mathbb{R}^n$; span, linear independence, basis, dimension, rank; inner products, norms,
orthogonality, orthogonal projection; Gram–Schmidt and the QR decomposition; determinants in $n$ dimensions as signed
volume; eigenvalues and eigenvectors; symmetric matrices and the spectral theorem; positive definiteness and Cholesky;
the SVD.

**Why.** Every identity in Section 2 of the geometry notes uses one of these. The Gram determinant *is* Gram–Schmidt;
PCA *is* the spectral theorem; the Cholesky factor in the manual's VI.7 *is* a volume decomposition.

**Resources.**

- 3Blue1Brown, *Essence of Linear Algebra* (YouTube, free). Start with the determinant and eigenvector episodes.
- Gilbert Strang, *Introduction to Linear Algebra*, with the MIT 18.06 lectures (MIT OpenCourseWare, free).
- Sheldon Axler, *Linear Algebra Done Right*, 4th edition (open access): for when you want the proofs.

**Project exercise.** In numpy, *from scratch*: Gram–Schmidt on the nine sector return vectors; check
$\det G = \prod r_{ii}^2$; eigen-decompose the correlation matrix and plot the first eigenvector (the "market mode": it
should be roughly equal weights on every sector).

**Done when** you can prove $\|a_1 \wedge \cdots \wedge a_k\|^2 = \det(A^\top A)$ on paper via QR, and the code agrees.

### Stage 2 · The wedge product and geometric algebra · 2–3 weeks · late January 2027

**Learn.** The three axioms of the wedge product; $k$-vectors, blades and grade; the coordinates of a bivector in 3D and
their link to the cross product; the geometric product $ab = a\cdot b + a\wedge b$; duality; (optional) rotors.

**Resources.**

- Mathoma, *Geometric Algebra* series (the one the video recommends).
- sudgylacmoe, *A Swift Introduction to Geometric Algebra* (YouTube).
- Alan Macdonald, *Linear and Geometric Algebra*: undergraduate level, written for exactly this step.
- Dorst, Fontijne and Mann, *Geometric Algebra for Computer Science*: deeper, if you enjoy it.

**Project exercise.** For three vectors $a, b, c \in \mathbb{R}^4$, compute $\|a\wedge b\wedge c\|^2$ as the sum of the
squares of the four 3×3 minors of $[a\;b\;c]$ (the Cauchy–Binet formula) and check it against $\det G$. This is the wedge
product computed *without* a Gram matrix: it shows the two definitions agree.

**Done when** you can explain why the cross product, read as the normal to a plane, works only in 3D while the wedge
product works in $\mathbb{R}^{60}$.

### Stage 3 · Multivariate statistics · 6–8 weeks · February to April 2027 (Phase 2)

**Learn.** Random vectors; the mean vector and covariance matrix (why it is symmetric and positive semi-definite); the
correlation matrix; the multivariate normal; linear regression as orthogonal projection (the hat matrix); $R^2$, partial
correlation, the variance inflation factor; generalised variance ($\det\Sigma$); PCA as a statistical method; the sampling
error of a correlation matrix when the window $W$ is not much larger than $n$; shrinkage (Ledoit–Wolf).

**Why.** This is where geometry meets data. It is also exactly Phase 2 of your plan: **week 20 is "Multicollinearity: VIF
and correlation clustering"**, and identity I5 in the geometry notes says VIF *is* a leave-one-out volume.

**Resources.**

- James, Witten, Hastie and Tibshirani, *An Introduction to Statistical Learning* (statlearning.com, free): the linear
  regression and PCA chapters.
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*: covariance matrices, generalised variance, PCA.
- Gilbert Strang, *Linear Algebra and Learning from Data*: the bridge from Stage 1 to machine learning.

**Project exercise.** The ML-redundancy hook from the geometry notes (Section 4.9): compute $\sqrt{\det R}$ of the 30
standardised ML features and rank every feature by $1/\sqrt{\mathrm{VIF}}$. Then rebuild the Monte Carlo null for
$W = 60$, $n = 9$ and reproduce the 0.73.

**Done when** you can derive $\det R = \prod_i (1 - R^2_{\,i\mid 1..i-1})$ and explain what it means for a portfolio.

**May and June 2027: exams.** Reading only, as the manual says. Good reading: Rabiner's HMM tutorial (Stage 4), and the
geometry notes' "what to doubt" list.

### Stage 4 · Time series, Markov chains and regimes · 6–8 weeks · July to September 2027

**Learn.** Stationarity (strict and weak), autocorrelation, AR/ARMA, volatility clustering and GARCH; Markov chains
(transition matrices, stationary distributions, expected holding times); hidden Markov models: the forward algorithm,
backward smoothing, Viterbi and Baum–Welch; regime-switching models; the difference between filtered and smoothed
probabilities, and why only filtered ones are allowed in a backtest.

**Resources.**

- Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", *Proceedings of the IEEE*.
- Hamilton, *Time Series Analysis* (1994), chapter 22 (changes in regime).
- Tsay, *Analysis of Financial Time Series*: the financial stylised facts and GARCH.
- Your QMUL Time Series module (MTH6139), if you take it: it covers the first half of this stage.

**Project exercise.** Implement the forward algorithm for a two-state Gaussian HMM in numpy (no new packages), then the
backward pass. Plot filtered against smoothed crisis probabilities for 2020: the smoothed curve "knows" the crash early,
which is the look-ahead the essay's comparison contains. Then run the fair comparison in Section 4.7–4.8 of the geometry
notes.

**Done when** you can write the forward recursion from memory and say why it is Bayes' theorem.

### Stage 5 · Put it to work · Phase 4 (22 July to 13 October 2027)

The geometry notes' Section 3, in order of value: (5) feature redundancy, (1) the live-mode structure row, (3) the
duplicate-window detector, then (2) the geometric regime label for the self-correction, and (6) effective bets in the
risk work of manual VI.7. Each produces an artefact, so each can go into `Dream/evidence_log.md`.

---

## 3. One page, for the wall

| When | Stage | Core idea | Proof that you have it |
|---|---|---|---|
| Now (1–2 wk) | 0 | FP1 already contains the 3D wedge product | Cross-product area equals $\sqrt{\det G}$ on real returns |
| Nov–Jan | 1 | Volume = product of Gram–Schmidt heights | Your own Gram–Schmidt matches `det` |
| Late Jan | 2 | The wedge product works in any dimension | Cauchy–Binet check in $\mathbb{R}^4$ |
| Feb–Apr | 3 | Volume = $\det R$ = product of $(1-R^2)$ = 1/VIF chain | Ranked redundancy of the 30 ML features |
| May–Jun | — | Exams: reading only | — |
| Jul–Sep | 4 | Regimes, HMMs, filtered vs smoothed | Your own forward algorithm, and the fair comparison |
| Phase 4 | 5 | Geometry inside the project | Evidence-log entries with artefacts |

---

*If a stage takes longer than planned, cut Stage 2's optional rotors and Stage 4's Viterbi first. Never cut the project
exercise at the end of a stage: that is the part that turns reading into knowing.*
