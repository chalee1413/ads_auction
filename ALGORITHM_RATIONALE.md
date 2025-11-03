# Algorithm Rationale

Quick reference guide explaining why each algorithm was chosen for incrementality measurement. Algorithms were tested on real data, compared, and documented.

**Summary**: Algorithms differ in performance and suitability. Each algorithm works better for specific use cases and has different trade-offs.

For design decisions and implementation architecture, see "DESIGN_DECISIONS.md".

---

## Table of Contents

1. [Core Metrics](#1-core-metrics)
2. [Variance Reduction](#2-variance-reduction)
3. [Causal Inference](#3-causal-inference)
4. [Uplift Modeling](#4-uplift-modeling)
5. [Experimentation Methods](#5-experimentation-methods)
6. [Auction Theory](#6-auction-theory)
7. [Data Processing](#7-data-processing)

---

## 1. Core Metrics

### 1.1 Incremental Return on Ad Spend (iROAS)

**Algorithm**: "iROAS = Incremental Revenue / Ad Spend"

**Why iROAS over ROAS?**
- **ROAS**: Attributes all conversions to ads (over-attribution)
  - Example: Brand loyalists would convert anyway
  - Data shows: 1,654% ROAS vs 39% iROAS (1,615 percentage point gap)
- **iROAS**: Measures true causal effect (test vs control)
  - Isolates incremental revenue that wouldn't exist without ads
  - Data shows: Only $279 of $11,729 attributed revenue is incremental

**Why This Formula?**
- Standard industry formula (Remerge, Lifesight, IMM Guide)
- Percentage format (*100) matches industry conventions
- Directly actionable: iROAS > 100% = profitable, < 100% = reallocate

**Alternatives Considered:**
- **ROAS**: REJECTED - Over-states effectiveness by 40x
- **CPA (Cost Per Acquisition)**: REJECTED - Doesn't capture value heterogeneity
- **LTV/ROAS**: REJECTED - Too complex, iROAS sufficient for measurement
- **iROAS**: SELECTED - Correctly isolates incremental value

**References**: 
- Remerge: https://www.remerge.io/findings/blog-post/a-quick-guide-to-interpreting-incremental-roas
- Lifesight: https://lifesight.io/glossary/what-is-incremental-roas-iroas/
- IMM Guide: https://imm.com/blog/iroas-incremental-return-on-ad-spend

**Implementation**: "demo/incrementality.py" - "calculate_iroas()"

---

### 1.2 Lift Calculation

**Algorithm**: "Lift = (Test - Control) / Control * 100%"

**Why This Formula?**
- Standard percentage lift calculation
- Industry convention for reporting lift
- Handles edge cases (zero control)

**Limitations:**
- Can be misleading when control is near zero (e.g., 479% lift not significant in some datasets)
- Always report absolute effects alongside percentages

**Implementation**: "demo/incrementality.py" - "calculate_lift()"

---

## 2. Variance Reduction

### 2.1 CUPED (Controlled-experiment Using Pre-period Data)

**Algorithm**: 
- "Theta = Cov(pre, post) / Var(pre)"
- "Adjusted effect = Effect - Theta * (Pre-period diff)"
- "Variance reduction = 1 - (Theta^2 * Var(pre) / Var(post))"

**Why CUPED?**
- **Statistical Power**: Reduces variance by 10-30%
  - Detects smaller effects with same sample size
  - Or same detection with smaller samples (cost savings)
- **Industry Standard**: Microsoft Research 2013, widely adopted
  - Used by Meta, Google, Microsoft for variance reduction
- **Cost Efficiency**: Smaller sample sizes reduce experiment costs

**When to Use:**
- A/B tests with pre-period baseline data
- High pre-post correlation (user-level metrics, revenue)
- Sample size constraints or cost-sensitive experiments

**When NOT to Use:**
- No pre-period data available
- Pre-post correlation is low (limited benefit)
- Already well-balanced randomized experiments

**Why Combined Theta?**
- More stable estimate than separate Thetas for test/control
- Standard practice in CUPED literature
- Assumes treatment doesn't affect pre-post correlation (reasonable)

**Alternatives Considered:**
- **No adjustment**: REJECTED - Wastes statistical power
- **Stratification**: REJECTED - Good for categorical, but CUPED handles continuous better
- **Regression adjustment**: REJECTED - More complex, CUPED is simpler and proven
- **CUPED**: SELECTED - Simple, proven, industry standard

**References**: Microsoft Research 2013 CUPED methodology

**Implementation**: "demo/cuped.py" - "cuped_adjustment()"

---

## 3. Causal Inference

### 3.1 Tree-Based Causal Inference (Wang et al. 2015)

**Algorithm**: 
- Build decision tree with treatment interaction features: "[X, treatment, X * treatment]"
- Split on treatment effect heterogeneity, not just outcomes
- Individual Treatment Effect: "ITE(x) = mu_1(x) - mu_0(x)" (counterfactual difference)

**Why Tree-Based Causal Inference?**
- **Heterogeneous Effects**: Treatment effects vary across segments (e.g., 12x difference observed in analysis)
  - Segment 7: Effect = 0.0660 (12x average, 2.35% of population)
  - Segment 15: Effect = 0.0388 (7x average, 4.06% of population)
- **Interpretability**: Trees provide clear segment descriptions
  - Example: "Users aged 25-35 in US with high purchase history"
- **Actionability**: Segments used for targeting, budget allocation, and creative optimization
- **Research Foundation**: Wang et al. (2015) WSDM designed for ad effectiveness

**Why Treatment Interaction Features?**
- "X * treatment" captures heterogeneous effects
- Tree splits on where treatment effects differ, not just where outcomes differ
- Key innovation from Wang et al. (2015)

**Why Decision Tree, Not Random Forest?**
- Single tree is more interpretable (one path = one segment)
- Clear segment descriptions for targeting/budget allocation
- Wang et al. (2015) uses single trees (can ensemble later if needed)
- Random forest would be less interpretable

**Alternatives Considered:**
- **Linear regression**: REJECTED - Cannot capture heterogeneous effects
- **Propensity Score Matching**: REJECTED - Handles confounding but doesn't identify heterogeneity
- **Meta-learners (T/X-learners)**: REJECTED - Good for individual effects but less interpretable segments
- **Random Forests**: REJECTED - Could work but Wang's causal tree is purpose-built
- **Causal Tree**: SELECTED - Purpose-built for heterogeneous treatment effects, interpretable

**References**: 
- Wang, P., Sun, W., Yin, D., Yang, J., Chang, Y. (2015). "Robust Tree-based Causal Inference for Complex Ad Effectiveness Analysis." WSDM '15, pages 67-76.

**Implementation**: "demo/tree_causal.py" - "CausalTree" class

---

### 3.2 Propensity Score Matching (PSM)

**Algorithm**: 
- Estimate propensity score: "e(x) = P(T=1|X=x)" using logistic regression
- Match treated units to control units based on propensity score similarity
- Average Treatment Effect on Treated (ATT): "E[Y₁ - Y₀ | T=1]"

**Why PSM?**
- **Observational Data**: Works when randomization isn't possible
- **Confounding Control**: Balances observed confounders via matching
- **Interpretability**: Clear matched pairs for analysis

**When to Use:**
- Observational data (no randomization)
- Need to control for observed confounders
- Want interpretable matched pairs

**When NOT to Use:**
- Randomized experiments (use RCT instead)
- Many unobserved confounders (PSM doesn't help)
- Large datasets (computationally expensive)

**Limitations:**
- Only controls for observed confounders
- Requires overlap: Both treated and control units with similar propensity scores
- Can be computationally expensive with many features

**References**: Standard causal inference methodology

**Implementation**: "demo/psm.py" - "PropensityScoreMatching" class

---

### 3.3 Difference-in-Differences (DID)

**Algorithm**: 
- "DID = (Y_post_treat - Y_pre_treat) - (Y_post_control - Y_pre_control)"
- Two-way fixed effects: Time and group fixed effects

**Why DID?**
- **Temporal Analysis**: Leverages pre-post data to control for trends
- **Unobserved Confounders**: Assumes parallel trends (same unobserved factors affect both groups)
- **Real-World Applicability**: Works when randomization isn't possible (geo experiments)

**When to Use:**
- Pre-post data available
- Parallel trends assumption holds
- Geo-based experiments (geo holdout tests)

**When NOT to Use:**
- Parallel trends assumption violated
- No pre-period data
- Randomized experiments (use RCT instead)

**Parallel Trends Assumption:**
- Treated and control groups would have followed same trend in absence of treatment
- Testable using pre-period data
- Critical assumption - must validate

**References**: Standard causal inference methodology

**Implementation**: "demo/did.py" - "difference_in_differences()"

---

## 4. Uplift Modeling

### 4.1 Meta-Learner Framework

**Why Multiple Learners?**
Each learner has different strengths and performs best in different scenarios.

#### T-Learner: Two Separate Models

**Algorithm**: 
- "mu_0(x)": Model trained on control group
- "mu_1(x)": Model trained on treatment group
- Treatment effect: "tau(x) = mu_1(x) - mu_0(x)"

**Why T-Learner?**
- Best when treatment and control groups have fundamentally different behavior
- Allows each group to have completely different model structure
- Simple and interpretable (two independent models)

**Limitations:**
- Requires sufficient data in both groups
- Can overfit if groups are small
- Doesn't leverage information about treatment assignment

**Performance**: MSE=0.0023, MAE=0.0390 (highest error in comparison)

**When to Use**: Large sample sizes in both groups, groups have very different characteristics

---

#### S-Learner: Single Model with Treatment Indicator

**Algorithm**: 
- Single model: "mu(x, w)" where "w" is treatment indicator
- Treatment effect: "tau(x) = mu(x, 1) - mu(x, 0)"

**Why S-Learner?**
- Most data-efficient (uses all data for one model)
- Best when treatment effect is additive (simple interactions)
- Often performs well in practice

**Performance**: MSE=0.0011, MAE=0.0331 (best performer in comparison)

**Limitations:**
- Assumes treatment effect is captured by single feature
- Can underfit heterogeneous effects if treatment isn't emphasized
- Treatment indicator might get low importance if outcome dominates

**When to Use**: Balanced or large datasets, treatment effects relatively uniform

---

#### X-Learner: Cross-Learner Algorithm

**Algorithm**: 
- Train models to impute missing counterfactuals
- Cross-imputation: Use control group to impute treated, and vice versa
- Combine imputations with propensity score weighting

**Why X-Learner?**
- Best when sample sizes are imbalanced (more control than treated)
- Handles imbalanced data better than T-learner
- Combines strength from both groups

**Performance**: MSE=0.0011, MAE=0.0333 (second best in comparison)

**When to Use**: Imbalanced treatment/control groups

---

#### DR-Learner: Doubly Robust Learner

**Algorithm**: 
- Combines outcome model and propensity model
- Doubly robust: Works if outcome model OR propensity model is correct
- More robust than single-model approaches

**Why DR-Learner?**
- Most robust (doubly robust property)
- Works even if outcome model OR propensity model is misspecified
- Best for production where model assumptions may not hold

**Performance**: MSE=0.0011, MAE=0.0334 (most robust approach)

**When to Use**: Production deployment, uncertainty about model assumptions

---

**Decision Rule**:
- **Speed**: Use S-Learner (single model, fastest)
- **Imbalanced Data**: Use X-Learner
- **Production Robustness**: Use DR-Learner
- **Different Group Behavior**: Use T-Learner

**Why Not Ensemble?**
- Could ensemble all four, but adds complexity without clear benefit
- Each learner already performs well on its use case
- Ensemble would require additional validation

**References**: 
- Kunzel et al. (2019). "Meta-learners for estimating heterogeneous treatment effects using machine learning."

**Implementation**: "demo/uplift_models.py" - Four separate classes

---

## 5. Experimentation Methods

### 5.1 Randomized Controlled Trials (RCT)

**Algorithm**: 
- Random assignment to treatment/control
- 50/50 split for equal sample sizes
- Statistical testing: Two-sample t-test

**Why RCT?**
- **Gold Standard**: Most reliable causal inference method
- **Unbiased**: Eliminates selection bias through randomization
- **Statistical Rigor**: Clear statistical tests and confidence intervals

**When to Use:**
- Feasible to randomize (users, impressions, campaigns)
- Need highest causal validity
- Can afford control group (revenue loss)

**When NOT to Use:**
- Cannot randomize (geo experiments, market-level)
- Control group too expensive (use ghost bidding instead)
- Pre-existing data (use observational methods)

**Implementation**: "demo/experimentation.py" - "rct_analysis()"

---

### 5.2 Ghost Bidding (Moloco Methodology)

**Algorithm**: 
- Bid on impressions normally
- When winning, randomly assign some winners to "control" (don't serve ad)
- Track conversions for both served (test) and not-served (control)
- Difference gives incremental lift without holding out entire segments

**Why Ghost Bidding?**
- **Revenue Preservation**: No revenue loss from traditional holdout groups
- **Real-Time Measurement**: Continuous incrementality measurement in live auctions
- **Natural Control**: Same users, same auction participation, just no ad shown

**Why Not Traditional Holdout?**
- Traditional holdout: Hold out entire segments - revenue loss
- Ghost bidding: Participate in all auctions, just don't serve some ads - no revenue loss

**Limitations:**
- Requires access to raw bid streams and impression logs
- Our Kaggle datasets lacked sufficient bid data for full simulation
- Production implementation requires auction system integration

**When to Use:**
- Real-time incrementality measurement needed
- Cannot afford revenue loss from traditional holdout
- Have access to bid stream and impression logs

**References**: 
- Moloco blog: https://www.moloco.com/blog/proving-incremental-roas-retail-media-advertising

**Implementation**: "demo/ghost_bidding.py" - "ghost_bidding_simulation()"

---

### 5.3 Geo Holdout Experiments

**Algorithm**: 
- Randomly assign geographic regions to treatment/control
- Treatment: Show ads in treated geos
- Control: Hold out ads in control geos
- Measure difference in conversions between geos

**Why Geo Holdout?**
- **Cannot Randomize Users**: Some platforms don't allow user-level randomization
- **Natural Segmentation**: Geos are natural units for experiments
- **Large Sample Sizes**: Geos provide sufficient sample sizes for power

**When to Use:**
- Cannot randomize at user/impression level
- Need large sample sizes
- Geos are natural experiment units

**Limitations:**
- Requires sufficient geos (need power for geo-level analysis)
- Spillover effects between geos (contamination)
- Geographic differences (not just treatment effects)

**Implementation**: "demo/experimentation.py" - "geo_holdout_experiment()"

---

## 6. Auction Theory

### 6.1 Second-Price Auction Bidding

**Algorithm**: 
- **Second-price auction**: "bid = incremental_value_per_impression"
- **First-price auction**: "bid = incremental_value * (1 - shading_factor)"

**Why Second-Price Auction Strategy?**
- **Economic Theory**: In second-price (Vickrey) auctions, optimal strategy is to bid true value
- **Since we want incremental value, we bid incremental value**: Simple and optimal
- **No Bid Shading**: Unlike first-price auctions, no shading needed

**Why iROAS-Based Bidding, Not ROAS-Based?**
- **ROAS-Based**: Bids to attributed value - over-bids on non-incremental conversions
  - Example: ROAS = 1,654% - bids high - wastes budget on non-incremental
- **iROAS-Based**: Bids to incremental value - correctly values only incremental conversions
  - Example: iROAS = 39% - bids appropriately - allocates to incremental opportunities

**Why Not ROAS-Based?**
- ROAS-based bidding over-bids on non-incremental conversions
- Data shows: ROAS (1,654%) vs iROAS (39%) = 1,615 percentage point gap
- Analysis shows only $279 of $11,729 test revenue is incremental, demonstrating budget misallocation when optimizing on ROAS

**Why Not CPA-Based?**
- CPA doesn't account for value heterogeneity
- Some conversions are worth more (LTV varies)
- iROAS captures value, not just acquisition cost

**References**: 
- IMM Guide: https://imm.com/blog/iroas-incremental-return-on-ad-spend
- Vickrey auction theory: Optimal strategy in second-price auction is true value bidding

**Implementation**: "demo/auction.py" - "iroas_based_bidding()"

---

### 6.2 Budget Pacing

**Algorithm**: 
- Calculate pacing ratio: "pacing = spent / (budget * time_elapsed / total_time)"
- Target: "0.8 <= pacing <= 1.2" (80-120% of target pace)

**Why Budget Pacing?**
- **Budget Constraints**: Must spend full budget by end of period
- **Avoid Overspending**: Don't exhaust budget early
- **Avoid Underspending**: Ensure full budget utilization

**Why This Pacing Range (80-120%)?**
- 80% lower bound: Ensures sufficient spending to reach target
- 120% upper bound: Prevents exhausting budget too early
- Standard range for budget pacing to balance smooth spending and target achievement

**Implementation**: "demo/auction.py" - "budget_pacing()"

---

## 7. Data Processing

### 7.1 Sample Size: Why 10,000?

**Decision**: Use 10,000 samples per dataset

**Why 10,000?**
- **Computational Efficiency**: Full datasets (3M+ rows) too slow for rapid iteration
- **Statistical Power**: 10,000 samples sufficient for most analyses
  - 50/50 split: 5,000 test, 5,000 control
  - Actual detectable effect size depends on variance--sufficient sample size for meaningful analysis
- **Demonstration Purpose**: Proof-of-concept doesn't require full datasets
- **Scalability**: Code architecture supports millions - just increase sample size

**Alternatives Considered:**
- **Full datasets**: REJECTED - Too slow for iterative development
- **1,000 samples**: REJECTED - Insufficient statistical power
- **100,000 samples**: REJECTED - Unnecessary overhead for demo
- **10,000 samples**: SELECTED - Balance of efficiency and statistical power

**Production Scaling**:
- Framework supports full datasets via streaming/batch processing
- Same algorithms work on 10K or 10M samples
- Architecture (feature stores, distributed computing) handles scale

**Implementation**: "demo/data_processing.py" - "load_kaggle_data()" with sample size parameter

---

### 7.2 Feature Selection: Why Variance-Based?

**Decision**: Select features based on variance for RTB dataset (missing headers)

**Why Variance-Based?**
- **Missing Headers**: RTB dataset ("biddings.csv") lacks column headers
- **Variance Proxy**: High-variance columns likely contain outcome signals
  - Conversions, engagement typically have high variance
  - Static features have low variance
- **Robustness**: Works even without domain knowledge
- **Practical Necessity**: Required to proceed with analysis

**Limitations:**
- Spend estimation required (no actual bid prices in dataset)
- Ghost bidding cannot run (no bid columns)
- Real production would have actual bid prices from auction logs

**Alternatives Considered:**
- **All columns**: REJECTED - Too many features, curse of dimensionality
- **Domain expertise**: SELECTED - Use interpretable feature columns (site_id, geo_id, etc.)
- **Correlation-based**: REJECTED - Could work but direct use of feature columns is better
- **Random selection**: REJECTED - Completely arbitrary
- **Feature columns**: SELECTED - Use numeric feature columns directly

**Production Solution:**
- Actual bid prices from auction logs
- Feature engineering based on domain knowledge
- Automated feature importance from models

**Implementation**: "demo/main.py" - "prepare_data()" uses numeric feature columns directly

---

### 7.3 Treatment Assignment: Why Deterministic?

**Decision**: Use deterministic treatment assignment (index-based 50/50 split) for demos

**Why Deterministic?**
- **Reproducibility**: Deterministic assignment ensures reproducible results
- **Simplicity**: Easy to understand and verify
- **Demonstration**: Shows methodology clearly without randomness complexity
- **Validation**: Can verify implementation against expected results

**Production vs Demo:**
- **Demo**: Deterministic (index % 2) - for reproducibility
- **Production**: True randomization (random seed) - for causal validity

**Why Not Random in Demo?**
- Random assignment gives different results each run
- Harder to validate implementation
- Deterministic allows verification against expected results

**Alternatives Considered:**
- **True randomization**: REJECTED - Would use in production, but demos benefit from determinism
- **Stratified randomization**: REJECTED - Better for production but adds complexity to demo
- **Deterministic**: SELECTED - Clear, reproducible, appropriate for demonstration

**Implementation**: "demo/data_processing.py" - "create_treatment_assignment()" with "deterministic" flag

---

## Algorithm Selection Decision Tree

**Need to measure incrementality?**
- **RCT**: If randomization is feasible - Use randomized controlled trial
- **Ghost Bidding**: If randomization feasible but can't afford holdout - Use ghost bidding
- **Geo Holdout**: If user-level randomization not possible - Use geo holdout
- **Observational**: If randomization not possible - Use PSM, DID, or uplift models

**Need to reduce variance?**
- **CUPED**: If pre-period data available - Use CUPED adjustment

**Need to find heterogeneous effects?**
- **Tree-Based Causal**: If want interpretable segments - Use causal tree (Wang et al.)
- **Uplift Models**: If want individual treatment effects - Use meta-learners (T/S/X/DR)

**Need to optimize bidding?**
- **iROAS-Based**: If want incremental value - Use iROAS-based bidding (second-price)
- **ROAS-Based**: REJECTED - Over-states effectiveness by 40x

**Need to process data?**
- **10K Samples**: If demo/prototype - Use 10K samples
- **Full Datasets**: If production - Use full datasets with distributed processing
- **Variance Selection**: If missing headers - Use variance-based feature selection

---

## References

All detailed citations and references are in:
- "demo/CITATIONS.md": Complete bibliography
- "DESIGN_DECISIONS.md": Design decisions
- "TECHNICAL_REPORT.md": Methodology explanations

---

*This document provides quick reference for algorithm choices. For design decisions and implementation details, see "DESIGN_DECISIONS.md".*

