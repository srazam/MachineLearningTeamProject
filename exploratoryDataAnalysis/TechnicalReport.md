# Business Logic Hypothesis 

---

## Analysis of Interview Transcripts

This section analyzes both the existing expense reimbursement system and user interview transcripts from employees across departments. The goal is to identify behavioral patterns, perceived inconsistencies and opportunities for system improvement. 

In terms of analyzing the user interview transcripts from employees across departments, a series of patterns have been identified. Across transcripts it seemed that the behavior and incentives of the system were very particular. This included a per diem with a flat rate of $100 a day, and when it comes to overall spending a higher receipt does not yield a higher reimbursement. The system seems to be riddled with unexplainable penalties and caps, such as always sticking to around $800 for a lot of amounts offered back. This is not to be confused with the reward given for efficiency and anything too high or low in terms of mileage per day is equally penalized. This also brings into question the length of the trip, and typically short trips and long trips are penalized while 4-6 day trips earn the highest rates. 

Another observation from the employees are potential temporal restrictions. An individual can benefit depending on the time period within a quarter, as end-of-quarter submissions receive better reimbursements. This logic continues to extend to days of the week as well, with some reporting that Tuesday and Thursdays take favor while Friday has an 8% lower average reimbursement. Besides days of the week and quarterly period, a user, though not proved, suggests that even the lunar cycle is affected with new moons slightly outperforming full moons. 

With incentives and time periods in mind, the structure thresholds of trips were also brought up as potential factors for differing treatment. Some claim that there is a “sweet spot” with a 5 day trip, 180+ miles a day, and a per diem spend of less than $100. While this would be a good way to secure a heavy reimbursement, 8+ day trips with a high spend rate are penalized and awarded less. To conclude, it is specifically recommended to take 4-6 day trips because users claim they have the best outcomes overall. 

While these have been relatively straight forward rules to an extent, let’s discuss inconsistency and perceived randomness. Typically, even when two trips are virtually identical in spending, length and more, they are still subject to 5-10% differences in their reimbursement. Overall there is a lot of variability and distrust from users because of this. The size of receipts also influence the amount awarded, with smaller or perhaps even none, allowing for the user to access higher per diem rates. 

Overall, users mainly have complaints towards the lack of transparency and fairness. Often highlighting how the system is dependent on user experience to better strategize or “game” the system. This creates a lot of frustration for new users or non-optimizers. 

## Proposed Business Rules and Logic Patterns 
The following business rules and logic patterns are derived from both transcript analysis and system behavior observed in historical data. While some patterns appear deliberate, others seem to result from legacy quirks or unintended artifacts in the original implementation.

Per Diem Ceiling Rule: 
- A fixed per diem rate of $100/day is applied, but users spending less than this threshold often receive higher reimbursements, implying an efficiency bonus. Spending more than this threshold does not increase reimbursement and may even decrease it.

Trip Duration Bands:
- Trips lasting 4–6 days receive the highest reimbursements.
- Trips shorter than 3 days or longer than 7 days receive penalized rates.
- A soft cap seems to apply beyond 8 days regardless of efficiency or miles traveled.

Mileage Efficiency Bonus:
- Ideal range: 180–220 miles/day. Users within this range appear to receive optimal results.
- Below or above this mileage window leads to diminishing returns.
- There may be a nonlinear scaling penalty for exceeding thresholds (~250+ miles/day).

Receipt-Based Adjustments:
- Smaller total receipt amounts (or no receipts at all) may result in higher per diem application, suggesting that the system sometimes favors minimal receipts.
- Larger receipts do not lead to proportional increases and may instead trigger soft caps or penalties.

Temporal Variability Factors:
- End-of-quarter submissions tend to yield higher reimbursement.
- Tuesday and Thursday submissions outperform others; Friday has an ~8% drop in average reimbursements.
- Lunar cycle effects are unconfirmed but noted in user perception—possibly noise or unexplained timing artifacts.

Inconsistency Threshold:
- Identical trips may vary by 5–10% in reimbursement due to undocumented internal variability—likely a combination of rounding, bugs, or batch timing artifacts.
- These inconsistencies must be preserved in the replica system to maintain fidelity.

Fixed Ceiling Effect:
- A mysterious cap appears to exist around $800, which many trips converge towards, despite variance in inputs.
- This could reflect a hardcoded maximum reimbursement or a diminishing return function kicking in.

---

## Feature Importance Hypotheses 
Based on the transcript patterns and observed behavior in historical factors, we hypothesize the following hierarchy of feature importance: 
- Trip Duration – Primary driver of reimbursement banding. Determines eligibility for bonuses or penalties.

- Miles per Day (Derived Feature) – Efficiency calculation based on total miles divided by days. Strong indicator of favorable reimbursement zones.

- Receipt Total – Nonlinear influence. Smaller totals may be over-rewarded; larger totals encounter penalties or flat caps.

- Trip Timing – Submission timing (day of week, quarter) appears to nudge outcomes slightly—secondary effect.

- Receipt Count (Implied from Total) – Minimal or no receipts seem to be overcompensated, suggesting the system prefers simplicity or defaults to per diem.

---

## Potential Non-Linear Relationships Identification 
A number of non-linear relationships are suspected based on the transcript abnormalities: 
- Diminishing returns on spending, where reimbursemtns seem to increase with recipts only up to a point, and then declines or plateus. This indicates a nonlinear cap or penalty function. 
- Ideal mileage per day seems to follow a bell curve with peak reimbursemnt in the 180-220 range, where reimbursements taper off sharply outside of it.
- Rewards increase sharply around day 3-4, peak at 5-6 days, then decline.
- Temporal modifiers also indicate non linear tendancies. 
- There is evidence of noise not dependant on user input, which could stem from rounding errors as mentioned in the transcript. 