# Results Explanation

How to interpret results from the incremental advertising measurement demo.

## Assumption Checks

All demos now include assumption checks that validate data requirements and statistical assumptions before computing results. Each demo displays:

- **PASS/FAIL status** for each assumption
- **Specific errors** if assumptions fail
- **WARNING messages** when assumptions fail (results may be invalid)

**Why Assumption Checks Matter:**
- Invalid results can occur when assumptions are violated
- Assumption checks help identify data quality issues
- Failed checks indicate results may be unreliable

**Common Assumptions:**
- **Sample Size**: Minimum sample sizes required (varies by algorithm)
- **RCT Balance**: Treatment/control groups should be roughly balanced (50/50 split)
- **Positivity**: Both treated and control groups must exist
- **Pre-Post Correlation**: CUPED requires high correlation between pre/post periods
- **Valid Data**: Spend, revenue, and bid data must be non-zero and valid

## Understanding the Numbers

**Key Context:**
- All conversion rates are probabilities (0.001 = 0.1% = 1 in 1,000)
- All comparisons are relative to the **control group** (no ads)
- Metrics show the **difference** between test (with ads) vs control (no ads)

**Units Explained:**
- **Conversion Rates (0.0010, 0.0018)**: Probability of conversion per user/impression (0.1% = 1 in 1,000, 0.18% = 1.8 in 1,000)
- **Treatment Effects (-0.0012)**: Change in conversion probability (absolute difference, not percentage)
- **Revenue ($50, $90)**: Total revenue across all users in the group
- **Lift (-44.44%)**: Percentage change relative to control: (Test - Control) / Control * 100%
- **ROAS/iROAS (1.10%, -0.88%)**: Return as percentage of ad spend (100% = breakeven)

## DEMO 1: CUPED

**Output**:
```
Assumption Checks:
  Sample Size: PASS
  Pre-Post Correlation: PASS (r=0.XXX)
  Balanced Groups: PASS
  All Assumptions: PASS

Test: 0.0010, Control: 0.0018
Lift: -44.44%, P-value: 0.2848
```

**Assumption Checks**:
- **Sample Size**: PASS if >= 10 samples per group (test_pre, control_pre, test_post, control_post)
- **Pre-Post Correlation**: PASS if correlation > 0.3 (indicates CUPED will be effective)
- **Balanced Groups**: PASS if groups are roughly equal size (ratio > 0.5)
- **WARNING**: If assumptions fail, CUPED results may be invalid or provide minimal benefit

**Interpretation**:
- **Test: 0.0010 (0.1%)**: 1 in 1,000 users converted with ads
- **Control: 0.0018 (0.18%)**: 1.8 in 1,000 users converted without ads
- **Lift: -44.44%**: Treatment converted 44% fewer users relative to control (worse performance)
  - Calculation: (0.0010 - 0.0018) / 0.0018 * 100% = -44.44%
- **P-value: 0.2848**: Not significant (>0.05). Difference could be random chance
- **Action**: Cannot conclude treatment has meaningful effect

## DEMO 2: Tree-based Causal Inference

**Output**:
```
Assumption Checks:
  Sample Size: PASS
  RCT Balance: PASS
  Positivity: PASS
    - 1 constant features detected
  All Assumptions: PASS

Average Treatment Effect: -0.0012
Effect Variance: 0.0001
Heterogeneity: 11 segments identified
```

**Assumption Checks**:
- **Sample Size**: PASS if >= 100 samples (recommended minimum)
- **RCT Balance**: PASS if treatment ratio between 0.1 and 0.9 (roughly 50/50)
- **Positivity**: PASS if both treated and control groups exist
- **WARNING**: Constant features detected (will be ignored by tree)
- **WARNING**: If assumptions fail, causal estimates may be biased

**Interpretation**:
- **ATE: -0.0012**: Average user has 0.12 percentage point lower conversion probability with treatment
  - Example: If baseline is 0.18%, treatment reduces to 0.06% (absolute difference of -0.0012)
  - Relative change: -0.0012 / 0.0018 = -66.7% (treatment performs 66.7% worse on average)
- **Effect Variance: 0.0001**: Treatment effects vary relatively little across users (low heterogeneity)
- **11 segments**: Different treatment effects across user segments (some may benefit, others harmed)
- **Action**: Identify high-effect segments to target, negative-effect segments to avoid

## DEMO 3: Uplift Modeling

**Output**:
```
Assumption Checks:
  Sample Size: PASS
  RCT Balance: PASS
  Positivity: PASS
  Sufficient Per Group: PASS
  All Assumptions: PASS

T-Learner: Avg Uplift = -0.0005, MSE = 0.0011
S-Learner: Avg Uplift = -0.0001, MSE = 0.0000
X-Learner: Avg Uplift = -0.0005, MSE = 0.0004
DR-Learner: Avg Uplift = -0.0005, MSE = 0.0006
```

**Assumption Checks**:
- **Sample Size**: PASS if >= 100 samples total
- **RCT Balance**: PASS if treatment ratio between 0.1 and 0.9
- **Positivity**: PASS if both treated and control groups exist
- **Sufficient Per Group**: PASS if >= 50 samples in each group (T-Learner needs sufficient data)
- **WARNING**: If assumptions fail, uplift models may overfit or produce unreliable estimates

**Interpretation**:
- **Average Uplift (-0.0005, -0.0001)**: Estimated change in conversion probability per user
  - -0.0005 = 0.05 percentage point reduction (absolute difference)
  - Relative to control: If control is 0.18%, treatment is 0.13% (-27.8% relative)
- **MSE (lower is better)**: Prediction accuracy of treatment effect estimates
  - MSE = 0.0011 means average squared error is 0.11 percentage points
  - S-Learner performs best (MSE = 0.0000 = near-perfect predictions on this data)
- **Action**: Use S-Learner for production (best accuracy)

## DEMO 4: Ghost Bidding Simulation

**Output**:
```
Simulated 5000 impressions (simulated bids)
Treatment CVR: 0.0009, Control CVR: 0.0000
Incremental Conversions: 4.0
Lift: 0.00%, P-value: 0.5049
```

**Interpretation**:
- **Treatment CVR: 0.0009 (0.09%)**: 0.9 conversions per 1,000 impressions with ads
- **Control CVR: 0.0000 (0%)**: No conversions observed in control (could be sampling)
- **Incremental Conversions: 4.0**: Out of 5,000 impressions, treatment produced 4 additional conversions
  - Treatment: 5,000 * 0.0009 = 4.5 expected conversions
  - Control: 5,000 * 0.0000 = 0 expected conversions
  - Incremental: 4.5 - 0 = 4.0 conversions
- **Lift: 0.00%**: Cannot calculate (division by zero when control = 0)
- **P-value: 0.5049**: Not significant (>0.05). Cannot conclude treatment effect
- **Note**: Uses simulated bids (datasets lack actual bid prices)

## DEMO 5: iROAS vs ROAS Comparison

**Output**:
```
Assumption Checks:
  RCT Balance: PASS
  Sample Size: PASS
  Valid Spend: PASS
  All Assumptions: PASS

Test Revenue: $50.00, Control: $90.00
Incremental Revenue: $-40.00
iROAS: -0.88%, ROAS: 1.10%
Gap: 1.99 percentage points
```

**Assumption Checks**:
- **RCT Balance**: PASS if treatment ratio between 0.1 and 0.9
- **Sample Size**: PASS if >= 10 samples in each group
- **Valid Spend**: PASS if test group spend > 0
- **WARNING**: If assumptions fail, iROAS/ROAS calculations may be invalid

**Interpretation**:
- **Test Revenue: $50.00**: Total revenue from test group (with ads)
- **Control Revenue: $90.00**: Total revenue from control group (no ads)
- **Incremental Revenue: -$40.00**: Treatment actually reduced revenue by $40
  - Calculation: $50 - $90 = -$40 (ads hurt revenue in this example)
- **iROAS: -0.88%**: For every $1 spent on ads, incremental return is -$0.0088 (loss)
  - Calculation: -$40 / $4,545 (estimated spend) * 100% = -0.88%
  - Negative means ads reduce revenue (not profitable)
- **ROAS: 1.10%**: Attributes all $50 test revenue to ads (incorrect)
  - If spend was ~$4,545: $50 / $4,545 * 100% = 1.10%
  - ROAS ignores that $90 would have happened anyway (wrong metric)
- **Gap: 1.99 points**: ROAS (-0.88% vs 1.10%) differs by 1.99 percentage points
  - Real-world gaps can be 1,000+ points (ROAS shows 1,654% vs iROAS 39% = 1,615 point gap)
- **Action**: Use iROAS for incrementality measurement. ROAS can misallocate budget.

## DEMO 6: Complete Experiment Workflow

**Output**:
```
Test: 5000 samples, mean = 0.0010
Control: 5000 samples, mean = 0.0018
Lift: -44.44%, P-value: 0.2848, Significant: False
```

**Interpretation**:
- **Balanced groups**: 5,000 users each (50/50 split)
- **Test mean: 0.0010 (0.1%)**: Average 0.1% conversion rate with ads
- **Control mean: 0.0018 (0.18%)**: Average 0.18% conversion rate without ads
- **Lift: -44.44%**: Treatment converted 44% fewer users relative to control
  - Calculation: (0.0010 - 0.0018) / 0.0018 * 100% = -44.44%
- **Not significant (p = 0.2848)**: Cannot conclude treatment effect is real (could be random)
- **Action**: Increase sample size or run longer if effect exists

## Statistical Significance

- **p < 0.05**: Statistically significant (industry standard)
- **p >= 0.05**: Not significant (cannot conclude effect)
- **Lower p-value**: Stronger evidence of true effect
