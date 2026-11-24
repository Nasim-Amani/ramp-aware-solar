# Tutorial: how this project developed, and where it goes next

This tutorial walks through the project from the beginning: the data, the main ideas, each experiment (what we asked, what we did, what we found, what we decided), what the literature says, and the plan for the paper. It is written so that someone new to the project — or to ramps — can follow it.

**Contents**
1. The big picture
2. The data
3. Six ideas you need
4. The journey, step by step (data exploration → E1 → E1b → E2 → ramp-risk test)
5. What the literature says
6. The turning point: our main contribution
7. Where we go next (Steps 1–3)
8. Lessons learned about doing research
9. Map of the notebooks
10. Glossary

## 1. The big picture

**The problem.** When a cloud passes in front of the sun, the output of a solar plant can drop by half within seconds. Such sudden changes are called **ramps**. Grid operators and battery controllers must react to them, so it would be useful to forecast them.

**What we tried first.** Forecast solar irradiance 10–60 seconds ahead with an LSTM, and make the forecast "ramp-aware" by adding a term to the training loss.

**What we discovered.** The ramp-aware LSTM seemed to catch many more ramps — but looking closely, it mainly predicted many jumps up and down (a **zigzag**), and some of these jumps landed near real ramps **by luck**. The usual ramp score counts these lucky hits as success.

**Where this leads.** The paper will (1) show that the usual ramp score can be fooled, (2) propose a fair ramp score, and (3) test a training loss that rewards one real jump instead of wiggles.

```
 data exploration ──► E1/E1b: standard vs ramp-aware LSTM ──► E2: weight λ ──► ramp-risk test
        │                       │                                 │
   ramps are real,        ramp-aware loss "catches"         forecasts zigzag;
   fast, short,           more ramps, costs accuracy       part of the gain is luck
   come in groups                                                 │
                                                                  ▼
                                literature: generous scores can be fooled (other fields)
                                                                  │
                                                                  ▼
                       NEXT: Step 1 random guesser → Step 2 fair score → Step 3 better loss
```

## 2. The data

- **What:** 1-second measurements of solar irradiance (and PV module signals), 16–26 February 2013, at Sohar University, Oman (location confirmed with a clear-sky model).
- **Cleaning** (earlier notebooks): removed duplicates, subtracted a small sensor offset (2.7 W/m² at night), kept the 11 complete days and the daylight hours.
- **For the models:** averaged to **10-s steps**. This keeps the full size of the ramps (a whole cloud edge passes within about 10 s) while keeping the model small.
- **Split by whole days, in time order:** train 16–23 Feb, validation 24 Feb, test 25 Feb (broken clouds) and 26 Feb (clear).

## 3. Six ideas you need

**1. Persistence.** The simplest forecast: "in 10–60 s, irradiance will be the same as now". Every model must beat it to be useful. At very short horizons it is hard to beat.

**2. Skill.** How much better a model is than persistence: $\text{skill} = 1 - \text{RMSE}_{model}/\text{RMSE}_{persistence}$. 0 % = as good as persistence; negative = worse.

**3. Ramp.** In our work: a 10-s change of irradiance larger than **80 W/m²**. This threshold is the 95th percentile of the 10-s changes on the broken-cloud training days — so "ramp" means "one of the largest 5 % of changes".

**4. Ramp score** (Nouri et al., 2024). For every forecast path (now → +10 s → … → +60 s) we look for ramps in the forecast and in reality. A predicted ramp counts as "caught" if a real ramp in the same direction happens within ±10 s. From this we get:
- **recall:** share of real ramps that were caught,
- **precision:** share of predicted ramps that were real,
- **F1:** both combined.

**5. Loss vs score.** The **loss** is what the model tries to make small *during training*. The **score** is how *we* judge the model *after training*. Changing one does not change the other — and a model can learn to please the score in unexpected ways.

**6. The ramp term in the loss** (Mathieu et al., 2016). Our "ramp-aware" loss is MSE plus λ × (error in the *size* of each 10-s change). It pushes the model to predict changes as big as the real ones. λ sets how strong this push is (λ = 0 → plain MSE).

## 4. The journey, step by step

### 4.1 Data exploration — `Ramps_analysis.ipynb`
**Question:** what do ramps look like in this data, and are they real?

**What we did:** day statistics (Variability Index, time with the sun blocked), ramps per hour, size of changes at 1 s / 10 s / 60 s, ramp events, a check with a second sensor, and how predictable ramps are from the past.

**What we found:**
- **Three types of days:** 5 clear, 2 with few clouds, 4 with broken clouds. 98 % of all ramps are on the broken-cloud days.
- **Ramps are real:** an independent sensor (PV module 2) moved the same way at **98.8 %** of the ramp steps.
- **Ramps are fast and short:** 72 % are completed within one 10-s step; the sun stays blocked for a median of only **20 s**.
- **Ramps come in groups:** after a calm period, a ramp follows in the next minute only **3 %** of the time; after many recent ramps, **55 %**.
- **But timing is not predictable:** the direction of the next change cannot be predicted from past changes.

**Decision:** model resolution 10 s; horizon 10–60 s; ramp threshold 80 W/m² per 10 s.

---

### 4.2 E1 / E1b — standard vs ramp-aware LSTM (3 seeds)
**Question:** does the ramp term help? And should the LSTM predict the irradiance values, or the change from the current value?

**What we did:** a simple LSTM (one layer, 64 units; input: last 10 min) in four setups: target (value / change) × loss (MSE / ramp-aware, λ = 1). Each setup trained with 3 random seeds.

**What we found (broken-cloud test day, 1500 ramp steps):**
- **MSE LSTM:** about 4 % better than persistence in RMSE, but **caught only 5 ± 3 ramps** — it smooths them away.
- **Ramp-aware LSTM:** **caught 178 ± 44 ramps** (35 times more), with many false alarms (precision 0.26), and lost its accuracy advantage over persistence.
- **Persistence had the lowest MAE** of all models; the LSTMs only won in RMSE, by avoiding the largest errors.
- **Rises (sun returns) were caught about twice as often as drops (cloud arrives).**
- Predicting the change instead of the value did not help.

**Decision:** keep the value target; the ramp-aware loss "works", but at a price — so tune λ next.

---

### 4.3 E2 — the weight λ of the ramp term
**Question:** is there a λ that keeps accuracy and still catches ramps?

**What we did:** λ = 0, 0.1, 0.3, 1, 3; compared accuracy and ramps caught; looked at whole forecast paths.

**What we found:**
- **A sharp trade-off:** more λ → more ramps caught, less accuracy. No λ gave both.
- **The key observation:** with the ramp term, forecast paths do **not** contain one clear jump. They contain **many medium jumps up and down (zigzag)**, e.g. 355 → 490 → 400 → 545 → 435 W/m². Some of these jumps happen to fall within ±10 s of a real ramp and count as "caught".

**Why it zigzags:** the ramp term compares predicted and real steps *position by position*. If the model does not know *when* the cloud will come, the safest answer is a medium jump at every position.

**What this means:** part of the "improvement" in ramp score is **luck**, not real foresight.

Figure 1 below shows the E2 results and the drops-vs-rises comparison.

![Figure 1. Our E2 results: (a) more ramps caught costs accuracy; (b) the models mostly catch the sun returning, not the cloud arriving.](tutorial_fig1_E2_results.png)

*Figure 1. Our E2 results: (a) more ramps caught costs accuracy; (b) the models mostly catch the sun returning, not the cloud arriving.*

---

### 4.4 Ramp-risk test — `E3_Ramp_Risk_Test.ipynb`
**Question:** if the *timing* of a ramp cannot be predicted, can its *chance* be predicted?

**What we did:** instead of 6 values, the model outputs **one probability**: the chance of a ramp in the next 60 s. Compared with three simple methods: a constant, a "recent ramps" lookup rule, and a logistic regression on three variability features.

**What we found:**
- **The chance of a ramp is clearly predictable** (ROC-AUC up to 0.93; reliable probabilities).
- **The simple logistic regression beat the LSTM.**
- The risk rises only **after** the first ramp of a restless period: the models know that a restless period *continues*, not that it is about to *start*.

**What this means:** past irradiance can tell *whether* ramps are likely, not *when*. This supports the explanation for the zigzag: the model cannot know the timing, so it hedges.

## 5. What the literature says

**1. The trade-off is old news in wind power.** Reviews (Gallego-Castillo et al., 2015) report that tuning a model for average error and tuning it for ramps "could not both be satisfied at once". A recent review of wind and solar ramps (Zhang et al., 2025) says there is **still no agreed way to measure ramp forecasts**, and calls for ramp-specific metrics and losses.

**2. Many researchers forecast the chance of a ramp.** Because timing is so hard, wind ramps are often forecast as probabilities (Cui et al., 2017; Bossavy et al.).

**3. Solar ramp scores are new and spreading.** Nouri et al. (2024) proposed the ramp score we use; benchmarks such as SolarBench (2026) now report ramp detection.

**4. Generous scores can be fooled — shown in another field.** In anomaly detection, Kim et al. (2022, AAAI) showed that **random guesses could reach top scores** under a generous scoring rule, and advised always comparing with a random guesser. Tatbul et al. (2018, NeurIPS) proposed event scores where one real event can be matched to only one prediction.

**5. Losses against smoothing exist** (gradient difference loss, DILATE, TILDE-Q, frequency losses), and Subich et al. (2025) compare the *size* of features regardless of their *position*.

**The gap:** nobody has shown that a ramp-aware **loss** can fool a ramp **score** through zigzag forecasts.

## 6. The turning point: our main contribution

> *Ramp forecasts must be evaluated and trained fairly. We show that a common ramp-aware loss can fool a common ramp score through zigzag forecasts, propose a fair way to score ramp forecasts, and test a simple loss that rewards one real jump instead of many wiggles.*

The illustration below (made-up numbers) shows the three kinds of forecast: (a) MSE spreads the drop out; (b) the current ramp term produces medium jumps everywhere, some landing near the real drop by luck; (c) the goal: one jump of the right size, even if slightly early.

![Figure 2. Illustration with made-up numbers: one real drop and three kinds of forecast (MSE smoothing, zigzag, one clear jump).](tutorial_fig2_zigzag_illustration.png)

*Figure 2. Illustration with made-up numbers: one real drop and three kinds of forecast (MSE smoothing, zigzag, one clear jump).*

## 7. Where we go next

The first paper is kept small and safe: three steps on our own dataset.

### Step 1 — Can a forecast that knows nothing get a good ramp score? (`E4_Fair_Ramp_Score.ipynb`)
- Make a **random guesser**: persistence plus random jumps along the forecast path. It knows nothing about the sky.
- Score it with the current ramp score and compare with our trained models.
- *If the random guesser scores close to the ramp-aware LSTM, the score can be fooled.*
- **No new training:** uses the saved forecasts of E1/E2 (or retrains the few needed if those files are lost).

### Step 2 — A fair ramp score (same notebook)
Score all forecasts again with rules that do not reward luck:
1. **one prediction per real ramp** — extra predictions count as false alarms;
2. **right direction** — a drop only matches a drop;
3. **zigzag index** — how often forecast paths change direction, compared with reality;
4. **first vs later ramps** — the first ramp after a calm period (a real warning) scored separately.

*Expected:* the ramp-aware model's advantage shrinks; the random guesser falls close to zero.

### Step 3 — A loss that rewards one real jump (`E5_Sorted_Loss.ipynb`)
- **Sorted-step loss:** before comparing predicted and real step sizes, sort both from largest to smallest. The *position* of a jump no longer matters, only its *size* — so a zigzag is penalised, but one jump slightly too early is not.
- **Extra setup:** a second output that gives the chance of a ramp in the next 60 s, trained together with the forecast (idea of Ding et al., 2019). Kept separate so each change can be judged on its own.
- Everything judged with the fair score from Step 2.

### Decision points
- **After Step 1:** if the random guesser scores well → the "score can be fooled" finding is confirmed; this alone carries the paper.
- **After Step 3:** if the sorted-step loss works → the paper also proposes a better method; if not → the paper is an honest evaluation study.

### Left for later
Testing on a second 1-s dataset; using the forecasts for battery control.

(Full details — code locations and which idea comes from which paper — are in `Pre_E4_updated_RQ.ipynb`.)

## 8. Lessons learned about doing research

1. **Look at the forecasts, not only the numbers.** The zigzag — our main finding — was invisible in the tables and obvious in the forecast-path plot.
2. **Change one thing at a time.** Every experiment compared setups that differed in one choice only.
3. **Use several seeds.** Some single-seed results (e.g. "the change target is worse") did not hold with 3 seeds.
4. **Report more than one metric.** MAE and RMSE told different stories; so did recall and precision.
5. **Compare with simple baselines.** Persistence, a lookup rule and a logistic regression kept the LSTM results honest.
6. **Let the data suggest the question.** The paper's direction came from an unexpected result, not from the first plan.

## 9. Map of the notebooks

| Notebook | Role |
|---|---|
| `R_Q.ipynb` | first research plan and feature ideas |
| `Ramps_analysis.ipynb` | data exploration of ramps |
| `E1_…ipynb` | target × loss, first run (1 seed) |
| `E1b_Target_vs_Loss_3seeds.ipynb` | the same with 3 seeds; drops vs rises |
| `E2_Lambda_Sweep.ipynb` | weight λ of the ramp term; forecast paths (zigzag) |
| `E3_Ramp_Risk_Test.ipynb` | forecasting the chance of a ramp |
| `Pre_E4_updated_RQ.ipynb` | roadmap: literature, gap, Steps 1–3 with code locations and credits |
| `E4_Fair_Ramp_Score.ipynb` | *next:* random guesser and fair score |
| `E5_Sorted_Loss.ipynb` | *after that:* sorted-step loss and the risk output |

## 10. Glossary

| Term | Meaning |
|---|---|
| **Ramp** | a 10-s change of irradiance larger than 80 W/m² |
| **Persistence** | forecast = current value; the reference every model must beat |
| **Skill** | improvement over persistence (0 % = equal, negative = worse) |
| **Precision** | share of predicted ramps that were real |
| **Recall** | share of real ramps that were predicted |
| **F1** | precision and recall combined (0–1) |
| **Tolerance (±10 s)** | how far in time a predicted ramp may be from a real one and still count |
| **Loss** | what the model minimises during training |
| **λ (lambda)** | weight of the ramp term in the loss |
| **Zigzag** | a forecast path that jumps up and down repeatedly |
| **Random guesser** | a fake forecast with random jumps; used to test whether a score rewards luck |
| **Seed** | the random start of training; different seeds give slightly different models |
| **ROC-AUC** | how well a probability separates "ramp coming" from "no ramp" (0.5 = guessing, 1 = perfect) |
| **Broken-cloud day** | a day with many passing clouds (VI above about 4) — where almost all ramps occur |
