# Design Decisions

Rationale for algorithm selection, code structure, and design trade-offs in the incrementality measurement system. Each section explains why specific approaches were chosen and what alternatives were considered.

---

## Table of Contents

1. [Core Algorithm Selection](#1-core-algorithm-selection)
2. [Data Processing Decisions](#2-data-processing-decisions)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Code Design Patterns](#4-code-design-patterns)
5. [Trade-offs and Alternatives](#5-trade-offs-and-alternatives)
6. [Performance Considerations](#6-performance-considerations)
7. [Production Readiness Choices](#7-production-readiness-choices)

---

## 1. Core Algorithm Selection

### 1.1 Why iROAS Instead of ROAS?

**Decision**: Incremental Return on Ad Spend (iROAS) was selected as the primary optimization metric instead of ROAS.

**Problem**: Analysis of Kaggle datasets reveals a 1,615 percentage point gap between ROAS (1,654%) and iROAS (39%). This gap indicates significant budget misallocation when optimizing on ROAS.

**Reasoning**:
1. **Attribution Bias**: ROAS attributes all conversions to advertising, ignoring that many users would convert anyway (brand loyalists, organic searchers). Analysis shows only $279 of $11,729 attributed revenue is incremental--the remainder would occur without ads.
2. **Causal Validity**: iROAS measures causal effect by comparing test vs control groups, isolating revenue that wouldn't exist without advertising.
3. **Budget Optimization**: Analysis shows only $279 of $11,729 test revenue is incremental. iROAS-based bidding correctly allocates budget to incremental opportunities.
4. **Empirical Evidence**: The gap is measurable and quantifiable from data analysis.

**Alternatives Considered**:
- **ROAS**: Rejected due to attribution bias and non-causal nature
- **Attributed Revenue**: Rejected - same issues as ROAS
- **CPA (Cost Per Acquisition)**: Useful but doesn't capture value heterogeneity
- **LTV/ROAS**: More complete but requires complex modeling; iROAS is sufficient for measurement

**Implementation**: "incrementality.py" - "calculate_iroas()" function uses test-control difference to isolate incremental revenue.

---

### 1.2 Why CUPED for Variance Reduction?

**Decision**: Implement CUPED (Controlled-experiment Using Pre-period Data) for variance reduction in A/B tests.

**Reasoning**:
1. **Statistical Power**: CUPED reduces variance by leveraging pre-period covariate information, enabling detection of smaller effects with same sample size (or same detection with smaller samples).
2. **Standard Practice**: Microsoft Research 2013 methodology is widely adopted in industry for variance reduction.
3. **Interpretability**: The adjustment is mathematically well-founded (covariance-based) and interpretable.
4. **Cost Efficiency**: CUPED reduces variance by leveraging pre-period covariate information, enabling smaller sample sizes for the same statistical power (variance reduction depends on pre-post correlation).

**Formula Rationale**:
- "Theta = Cov(pre, post) / Var(pre)": Captures correlation between pre and post periods
- "Adjusted effect = Effect - Theta * Pre_diff": Removes variance due to baseline differences
- Variance reduction depends on pre-post correlation: higher correlation = more reduction

**When to Use**:
- A/B tests with pre-period baseline data
- High pre-post correlation (user-level metrics, revenue)
- Sample size constraints or cost-sensitive experiments

**When NOT to Use**:
- No pre-period data available
- Pre-post correlation is low (limited benefit)
- Randomized experiments where baseline imbalance is already minimal

**Alternatives Considered**:
- **No adjustment**: Rejected - wastes statistical power
- **Stratification**: Good for categorical covariates, but CUPED handles continuous better
- **Regression adjustment**: More complex, CUPED is simpler and proven
- **Stratified sampling**: Requires upfront design; CUPED works post-hoc

**Implementation**: "cuped.py" - "cuped_adjustment()" function implements the Microsoft Research formula.

---

### 1.3 Why Tree-Based Causal Inference?

**Decision**: Use Wang et al. (2015) tree-based causal inference for heterogeneous effect estimation.

**Reasoning**:
1. **Heterogeneous Effects**: Treatment effects vary across segments (12x difference observed). Tree-based methods identify these segments.
2. **Interpretability**: Trees provide intuitive segment descriptions ("users aged 25-35 in US with high purchase history").
3. **Non-linear Effects**: Decision trees capture complex, non-linear interactions between features.
4. **Actionability**: Segments can be directly used for targeting, budget allocation, and creative optimization.
5. **Research Foundation**: Wang et al. (2015) WSDM paper specifically designed for ad effectiveness analysis.

**Why Not Other Causal Methods?**:
- **Linear regression**: Cannot capture heterogeneous effects
- **Propensity Score Matching**: Handles confounding but doesn't identify heterogeneity
- **Meta-learners (T/X-learners)**: Good for individual effects but less interpretable segments
- **Random Forests**: Could work but Wang's causal tree is purpose-built for treatment effects

**Splitting Criteria**:
We use treatment-outcome interactions ("X * treatment") to identify splits where treatment effects differ, not just where outcomes differ. This is the key innovation from Wang et al. (2015).

**Alternatives Considered**:
- **Causal Forest (Wager & Athey)**: More sophisticated but complex; Wang et al. is simpler and proven
- **BART (Bayesian Additive Regression Trees)**: Bayesian approach, good but computationally expensive
- **Neural networks**: Non-interpretable; trees provide actionable segments

**Implementation**: "tree_causal.py" - "CausalTree" class builds trees with treatment interaction features.

---

### 1.4 Why Multiple Uplift Modeling Approaches?

**Decision**: Implement four meta-learners (T, S, X, DR) rather than choosing one.

**Reasoning**:
1. **Different Strengths**: Each learner has different assumptions and performs best in different scenarios:
   - **T-Learner**: Best when treatment and control are fundamentally different (e.g., different user behavior)
   - **S-Learner**: Best when treatment effect is additive and simple
   - **X-Learner**: Best when sample sizes are imbalanced (more control than treated)
   - **DR-Learner**: Most robust - combines outcome and propensity models (doubly robust)

2. **Model Selection**: Allows comparison and automatic selection of best-performing model per use case.

3. **Robustness**: DR-Learner provides robustness - works even if outcome model OR propensity model is misspecified (doubly robust property).

4. **Industry Standard**: These four represent the state-of-the-art meta-learner approaches (Kunzel et al. 2019).

**Performance on Data**:
- **S-Learner & X-Learner**: Lowest error (MSE: 0.0011, MAE: ~0.033)
- **DR-Learner**: Slightly higher error but more robust
- **T-Learner**: Highest error but useful when groups differ fundamentally

**Decision Rule**: Use S-Learner for speed, X-Learner for imbalanced data, DR-Learner for production robustness.

**Alternatives Considered**:
- **Single best model**: Rejected - no universal best; depends on data characteristics
- **Ensemble**: Could ensemble all four, but adds complexity without clear benefit
- **Neural uplift models**: State-of-the-art but complex and requires more data

**Implementation**: "uplift_models.py" - Four separate classes implementing each meta-learner.

---

### 1.5 Why Ghost Bidding for Control Groups?

**Decision**: Implement ghost bidding methodology for creating live control groups.

**Reasoning**:
1. **Revenue Preservation**: Traditional holdout groups reduce revenue by withholding ads. Ghost bidding allows measurement without revenue loss.
2. **Real-time Measurement**: Enables continuous incrementality measurement in live auction environments.
3. **Industry Practice**: Moloco and other programmatic platforms use this methodology.
4. **Natural Control**: Bidding but not serving creates a natural control - same users, same auction participation, just no ad shown.

**How It Works**:
1. Participate in auctions normally (bid on impressions)
2. When we win, randomly assign some winners to "control" (don't serve ad)
3. Track conversions for both served (test) and not-served (control) winners
4. Difference gives incremental lift without holding out entire segments

**Limitations**:
- Requires access to raw bid streams and impression logs
- Our Kaggle datasets lacked sufficient bid data for full simulation
- Production implementation requires auction system integration

**Alternatives Considered**:
- **Traditional holdout**: Rejected - loses revenue
- **Geo holdout**: Good for testing, but requires geographic randomization
- **Time-based holdout**: Works but limits measurement to specific periods

**Implementation**: "ghost_bidding.py" - Simulates ghost bidding when bid data available.

---

### 1.6 Why Second-Price Auction Bidding Strategy?

**Decision**: Use iROAS-based bidding in second-price auctions.

**Reasoning**:
1. **Economic Theory**: In second-price auctions (Vickrey), optimal strategy is to bid your true value. Since we want incremental value, we bid incremental value.
2. **Simple Strategy**: No bid shading needed in second-price auctions (unlike first-price).
3. **Industry Standard**: Most programmatic exchanges use second-price (or variants).
4. **Transparency**: Second-price auctions are more transparent and predictable.

**Bidding Formula**:
- **Second-price**: "bid = incremental_value_per_impression"
- **First-price** (if needed): "bid = incremental_value * (1 - shading_factor)"

**Why Not ROAS-Based Bidding?**:
ROAS-based bidding over-bids on non-incremental conversions, wasting budget. iROAS-based bidding correctly values only incremental conversions.

**Alternatives Considered**:
- **ROAS-based**: Rejected - over-bids on non-incremental conversions
- **CPA-based**: Doesn't account for value heterogeneity
- **Budget-constrained optimization**: More complex, iROAS bidding is simpler and effective

**Implementation**: "auction.py" - "iroas_based_bidding()" function implements second-price strategy.

---

## 2. Data Processing Decisions

### 2.1 Why 10,000 Sample Size?

**Decision**: Use 10,000 samples per dataset for analysis.

**Reasoning**:
1. **Computational Efficiency**: Full datasets (3M+ rows) are computationally expensive for rapid iteration.
2. **Statistical Power**: 10,000 samples provide sufficient power for analyses--5,000 test and 5,000 control enables detection of meaningful effects (actual detectable effect size depends on variance).
3. **Demonstration Purpose**: Proof-of-concept doesn't require full datasets.
4. **Scalability**: Code architecture supports scaling to millions - just increase sample size.

**Calculation**:
- For 50/50 split: 5,000 test, 5,000 control
- Actual power depends on variance and effect size--sufficient sample size for demonstration purposes
- Production would use full datasets for maximum statistical power

**Production Scaling**:
- Framework supports full datasets via streaming/batch processing
- Same algorithms work on 10K or 10M samples
- Architecture (feature stores, distributed computing) handles scale

**Alternatives Considered**:
- **Full datasets**: Rejected - too slow for iterative development
- **1,000 samples**: Rejected - insufficient statistical power
- **100,000 samples**: Could use but unnecessary overhead for demo

**Implementation**: "data_processing.py" - "load_kaggle_data()" loads 10K samples with option to scale.

---

### 2.2 Why Variance-Based Feature Selection?

**Decision**: Use numeric feature columns directly from Real-time Auction dataset (Dataset.csv).

**Reasoning**:
1. **Real revenue data**: The Real-time Auction dataset contains actual revenue (`total_revenue`) and impression data (`total_impressions`), enabling direct iROAS calculations without synthetic data.
2. **Feature columns**: Use numeric columns (site_id, geo_id, device_category_id, advertiser_id, etc.) as input features for machine learning models.
3. **Temporal features**: The dataset includes `date` column for temporal analysis and CUPED adjustment.
4. **Interpretable features**: Unlike PCA-transformed features, these columns are interpretable and map to real-world attributes.

**Limitations**:
1. **Spend estimation**: Dataset does not contain actual bid prices - spend is estimated from impressions using CPM ($2.00 per 1000 impressions).
2. **No bid columns**: Ghost bidding demo cannot run as dataset lacks bid_price or competition_bid columns.

**Production Solution**:
- Actual bid prices from auction logs
- Complete feature engineering based on domain knowledge
- Automated feature importance from models

**Alternatives Considered**:
- **All columns**: Rejected - too many features (89), curse of dimensionality
- **Domain expertise**: Preferred but unavailable (no headers/documentation)
- **Correlation-based**: Could work but variance is simpler proxy
- **Random selection**: Rejected - completely arbitrary

**Production Solution**:
- Proper data schemas with column documentation
- Feature engineering based on domain knowledge
- Automated feature importance from models

**Implementation**: "data_processing.py" - "select_features_by_variance()" handles missing headers case.

---

### 2.3 Why Deterministic Treatment Assignment?

**Decision**: Use deterministic treatment assignment (index-based 50/50 split) for demos.

**Reasoning**:
1. **Reproducibility**: Deterministic assignment ensures reproducible results for demos.
2. **Simplicity**: Easy to understand and verify.
3. **Demonstration**: Shows the methodology clearly without randomness complexity.

**Production vs Demo**:
- **Demo**: Deterministic (index % 2) - for reproducibility
- **Production**: True randomization (random seed) - for causal validity

**Why Not Random in Demo?**:
Random assignment would give different results each run, making it harder to validate implementation. Deterministic allows verification against expected results.

**Alternatives Considered**:
- **True randomization**: Would use in production, but demos benefit from determinism
- **Stratified randomization**: Better for production but adds complexity to demo
- **Propensity-based**: Would use for observational data, but we're simulating RCT

**Implementation**: "data_processing.py" - "create_treatment_assignment()" with "deterministic" flag.

---

## 3. Implementation Architecture

### 3.1 Why Modular Design?

**Decision**: Separate each algorithm into its own module/file.

**Reasoning**:
1. **Maintainability**: Each algorithm is independent, easier to modify/debug.
2. **Reusability**: Modules can be imported independently for different use cases.
3. **Testing**: Each module can be tested in isolation.
4. **Clarity**: Clear separation of concerns makes codebase easier to understand.
5. **Citations**: Each module documents its own citations and methodology.

**Module Structure**:
```
incrementality.py    # Core metrics (iROAS, lift, ROAS)
cuped.py            # CUPED variance reduction
tree_causal.py      # Tree-based causal inference
uplift_models.py    # Meta-learners (T/S/X/DR)
psm.py              # Propensity Score Matching
did.py              # Difference-in-Differences
experimentation.py   # RCT frameworks
ghost_bidding.py    # Ghost bidding simulation
auction.py          # Auction theory and bidding
```

**Alternatives Considered**:
- **Monolithic file**: Rejected - unmaintainable, hard to test
- **Object-oriented only**: Could work but functional approach is clearer for algorithms
- **Package structure**: Could organize further but current structure is sufficient

**Implementation**: Each algorithm in separate file with clear interface.

---

### 3.2 Why Functional + Class Hybrid?

**Decision**: Use functional functions for simple algorithms, classes for complex models.

**Reasoning**:
1. **Simplicity**: Simple calculations (iROAS, lift) are clearer as functions.
2. **State Management**: Complex models (trees, learners) need state - classes are natural.
3. **Scikit-learn Compatibility**: Using sklearn's BaseEstimator interface provides consistency.
4. **Flexibility**: Functions are easy to compose; classes encapsulate complexity.

**Pattern**:
- **Functions**: Pure calculations ("calculate_iroas", "calculate_lift")
- **Classes**: Stateful models ("CausalTree", "TLearner", "SLearner")

**Alternatives Considered**:
- **All functions**: Would work but loses scikit-learn compatibility
- **All classes**: Over-engineered for simple calculations
- **Current hybrid**: Balances simplicity and functionality

**Implementation**: Simple metrics as functions, models as classes.

---

### 3.3 Why NumPy/Pandas Over Other Libraries?

**Decision**: Use NumPy/Pandas for data manipulation, scikit-learn for ML.

**Reasoning**:
1. **Industry Standard**: NumPy/Pandas/scikit-learn are the de facto standard for data science.
2. **Performance**: NumPy is highly optimized C code, fast for numerical operations.
3. **Ecosystem**: Large ecosystem of libraries built on NumPy.
4. **Familiarity**: Most data scientists know these tools.
5. **Mature**: Stable, well-tested, production-ready.

**Alternatives Considered**:
- **PyTorch/TensorFlow**: Overkill for our algorithms; adds complexity
- **CuPy (GPU NumPy)**: Could use for scale but not needed for current scope
- **Polars**: Faster than Pandas but less mature, smaller ecosystem
- **Dask**: Needed for distributed computing but adds complexity

**When to Switch**:
- **Production scale**: Would add Dask/Spark for distributed processing
- **Deep learning**: Would use PyTorch for neural uplift models
- **Real-time**: Would use optimized C++ for latency-critical paths

**Implementation**: All code uses NumPy arrays and Pandas DataFrames.

---

## 4. Code Design Patterns

### 4.1 Why Type Hints?

**Decision**: Use Python type hints throughout codebase.

**Reasoning**:
1. **Documentation**: Type hints serve as inline documentation.
2. **IDE Support**: Better autocomplete and error detection.
3. **Catch Errors**: Static type checkers (mypy) catch type errors early.
4. **Clarity**: Makes function signatures self-documenting.

**Example**:
```python
def calculate_iroas(
    incremental_revenue: float, 
    ad_spend: float
) -> float:
```

**Alternatives Considered**:
- **No type hints**: Rejected - less clear, harder to maintain
- **Full mypy checking**: Could add but not critical for demo
- **Docstrings only**: Type hints are more precise and enforceable

**Implementation**: All functions include type hints for parameters and returns.

---

### 4.2 Why Docstrings Over Comments?

**Decision**: Use docstrings explaining algorithms, not just inline comments.

**Reasoning**:
1. **API Documentation**: Docstrings generate documentation automatically.
2. **Usage Examples**: Docstrings can include examples.
3. **Algorithm Explanation**: Better place for detailed reasoning than inline comments.
4. **Standard Practice**: Python standard (PEP 257).

**Docstring Structure**:
- **Description**: What the function does
- **Algorithm/Citation**: Which paper/methodology it implements
- **Args**: Parameter descriptions
- **Returns**: Return value descriptions
- **Examples**: Usage examples when helpful

**Alternatives Considered**:
- **Minimal comments**: Rejected - insufficient documentation
- **Separate docs only**: Docstrings are more accessible
- **Current approach**: Docstrings + strategic inline comments

**Implementation**: All functions have detailed docstrings.

---

### 4.3 Why Separate Citation File?

**Decision**: Maintain "CITATIONS.md" separate from code.

**Reasoning**:
1. **Centralized**: All citations in one place, easy to reference.
2. **Completeness**: Can include full bibliographic details.
3. **Accessibility**: Non-programmers can access citations.
4. **Maintainability**: Updates citations without touching code.

**Also Included In**:
- Module docstrings: Quick reference
- Function docstrings: Direct citations
- TECHNICAL_REPORT.md: Full explanation

**Alternatives Considered**:
- **Only in code**: Would work but harder to browse
- **Only in report**: Code loses traceability
- **Current approach**: Both places for different audiences

**Implementation**: "CITATIONS.md" + docstrings in code.

---

## 5. Trade-offs and Alternatives

### 5.1 Simplicity vs Sophistication

**Decision**: Implement proven, interpretable algorithms over cutting-edge but complex ones.

**Reasoning**:
1. **Production Readiness**: Simpler algorithms are easier to debug, maintain, and explain.
2. **Interpretability**: Business stakeholders need to understand and trust the methods.
3. **Proven**: These algorithms have been validated in industry (Microsoft, Meta, etc.).
4. **Baseline First**: Start simple, add complexity only if needed.

**What We Chose**:
- CUPED (simple, proven) over advanced variance reduction
- Tree-based causal inference (interpretable) over neural causal models
- Meta-learners (well-understood) over state-of-the-art but experimental methods

**When to Upgrade**:
- **More data**: Neural uplift models if we have millions of samples
- **Better performance needed**: Could add ensemble methods
- **Real-time constraints**: Would optimize specific paths

**Implementation**: All algorithms are well-established, not experimental.

---

### 5.2 Accuracy vs Speed

**Decision**: Prioritize correctness and clarity over micro-optimizations.

**Reasoning**:
1. **Correctness First**: Wrong answers fast are worse than right answers slightly slower.
2. **Premature Optimization**: Don't optimize until we have performance requirements.
3. **Readability**: Clear code is easier to optimize later if needed.
4. **Scope**: Demo doesn't have strict latency requirements.

**Optimizations Applied**:
- NumPy vectorization (already fast)
- Efficient data structures (Pandas for structured data)
- No premature optimization

**Production Optimizations** (not in demo):
- C++ for hot paths (<100ms requirements)
- Caching for repeated computations
- Batch processing for throughput
- Distributed computing for scale

**Implementation**: Code prioritizes clarity; optimizations documented for production.

---

### 5.3 Generalization vs Specificity

**Decision**: Implement general-purpose algorithms that work across datasets.

**Reasoning**:
1. **Reusability**: Same algorithms work for RTB, video ads, e-commerce.
2. **Validation**: If it works on multiple datasets, more trustworthy.
3. **Production**: Real systems need to handle diverse use cases.
4. **Flexibility**: General algorithms can be tuned per use case.

**Dataset-Specific Adjustments**:
- Feature engineering per dataset (but same algorithms)
- Different outcome metrics (revenue, engagement, conversions)
- Different spend proxies (bid price, creative duration, etc.)

**Alternatives Considered**:
- **Dataset-specific algorithms**: Would perform better but less reusable
- **Current approach**: General algorithms, dataset-specific features

**Implementation**: Algorithms are dataset-agnostic; features are dataset-specific.

---

## 6. Performance Considerations

### 6.1 Why Not Parallel Processing (Yet)?

**Decision**: Use sequential processing for demo; document parallel approach for production.

**Reasoning**:
1. **Simplicity**: Sequential code is easier to understand and debug.
2. **Scope**: Demo datasets (10K samples) don't need parallelization.
3. **Architecture**: Code structure supports parallelization when needed.
4. **Documentation**: TECHNICAL_REPORT.md explains parallel architecture.

**Production Parallelization**:
- **Data loading**: Parallel I/O for multiple datasets
- **Feature computation**: Parallel feature engineering
- **Model training**: Parallel hyperparameter search
- **Prediction**: Batch prediction in parallel

**When to Add**:
- Processing millions of samples
- Real-time inference requirements
- Multiple concurrent experiments

**Implementation**: Sequential code that can be parallelized in production.

---

### 6.2 Why Not GPU Acceleration?

**Decision**: CPU-only implementation; GPU acceleration documented for production.

**Reasoning**:
1. **Algorithm Fit**: Our algorithms (trees, linear models) don't benefit much from GPUs.
2. **Data Size**: Current datasets fit in CPU memory easily.
3. **Accessibility**: CPU-only code runs on any machine.
4. **Complexity**: GPU code adds deployment complexity.

**When GPUs Help**:
- Neural uplift models (deep learning)
- Large datasets (billions of samples)
- Real-time inference at extreme scale

**Implementation**: CPU NumPy/Pandas; GPU options documented in scaling plan.

---

## 7. Production Readiness Choices

### 7.1 Why Error Handling?

**Decision**: Include error handling for edge cases (division by zero, empty arrays, etc.).

**Reasoning**:
1. **Robustness**: Production code must handle edge cases gracefully.
2. **Debugging**: Clear error messages help identify issues.
3. **User Experience**: Better error messages than crashes.
4. **Best Practice**: Defensive programming prevents failures.

**Examples**:
- Division by zero checks ("if ad_spend == 0")
- Empty array handling ("if len(array) == 0")
- Invalid input validation

**Alternatives Considered**:
- **Minimal checks**: Would work but less robust
- **Current approach**: Complete but not over-engineered

**Implementation**: All functions check edge cases.

---

### 7.2 Why Reproducibility Features?

**Decision**: Include random seeds, deterministic behavior options.

**Reasoning**:
1. **Debugging**: Reproducible results make it easier to debug.
2. **Validation**: Can verify results match expected outcomes.
3. **Comparisons**: Same random seed enables fair algorithm comparisons.
4. **Production**: Reproducibility is critical for audits and debugging.

**Features**:
- Random seed parameters
- Deterministic treatment assignment option
- Version tracking for results

**Implementation**: "np.random.seed()" used throughout; seeds configurable.

---

## 8. Algorithm-Specific Rationale

### 8.1 Incrementality Metrics ("incrementality.py")

**Why These Formulas?**
- **iROAS = Incremental Revenue / Ad Spend**: Standard industry formula (Remerge, Lifesight, IMM guide)
- **Incremental Revenue = Test - Control**: Causal difference isolates incremental effect
- **Lift = (Test - Control) / Control**: Standard percentage lift calculation

**Why Two-Sample T-Test?**
- Standard for comparing two groups
- Handles unequal variances (Welch's t-test)
- Provides p-values and confidence intervals
- Well-understood by stakeholders

**Why Compare ROAS vs iROAS?**
- Demonstrates over-attribution problem
- Quantifies measurement error from attribution
- Key insight for business stakeholders

---

### 8.2 CUPED ("cuped.py")

**Why This Formula?**
- **Theta = Cov(pre, post) / Var(pre)**: Standard CUPED coefficient (Microsoft Research 2013)
- Captures pre-post correlation
- Normalizes by pre-period variance

**Why Adjust Both Groups?**
- CUPED adjusts test and control to remove baseline differences
- More powerful than just adjusting difference
- Standard CUPED methodology

**Why Combined Theta?**
- Calculate Theta from combined test+control data
- More stable estimate than separate Thetas
- Standard practice in CUPED literature

---

### 8.3 Tree-Based Causal Inference ("tree_causal.py")

**Why Treatment Interaction Features?**
- "X * treatment" captures heterogeneous effects
- Allows tree to split on where treatment effects differ
- Key innovation from Wang et al. (2015)

**Why Separate Predictions?**
- Predict with treatment=1 and treatment=0 separately
- Difference gives Individual Treatment Effect (ITE)
- Standard causal inference approach

**Why Decision Trees, Not Random Forest?**
- Single tree is more interpretable
- Segments are clearer (one path = one segment)
- Wang et al. (2015) uses single trees
- Can ensemble later if needed

---

### 8.4 Uplift Models ("uplift_models.py")

**Why Meta-Learner Framework?**
- Flexible - can use any base learner
- Well-theoretically grounded
- Industry standard (Kunzel et al. 2019)

**Why Four Learners?**
- Different assumptions and strengths
- Allows comparison and selection
- DR-Learner provides robustness

**Why Separate Classes?**
- Clear separation of implementations
- Easy to test individually
- Can use independently

---

## 9. Future Enhancements and When to Add

### 9.1 Advanced Methods (Not Yet Implemented)

**Neural Uplift Models**:
- **When**: If we have millions of samples and complex interactions
- **Why Not Now**: More complex, harder to interpret, needs more data

**Causal Forests**:
- **When**: Need more robust heterogeneous effect estimation
- **Why Not Now**: Wang et al. tree is sufficient and simpler

**Synthetic Control**:
- **When**: Geo-based experiments with few treated units
- **Why Not Now**: Less general than our current methods

**Instrumental Variables**:
- **When**: Observational data with unobserved confounding
- **Why Not Now**: RCT data doesn't need IV

---

### 9.2 Production Infrastructure (Documented, Not Implemented)

**Feature Store**:
- **Why**: Consistent features across training/inference
- **When**: Multiple models, real-time inference
- **Current**: Features computed on-the-fly (fine for demo)

**Model Serving**:
- **Why**: Low-latency inference (<100ms)
- **When**: Real-time bidding integration
- **Current**: Batch prediction (fine for analysis)

**Experiment Platform**:
- **Why**: Manage multiple concurrent experiments
- **When**: Running production experiments at scale
- **Current**: Manual experiment setup (fine for demos)

**Monitoring**:
- **Why**: Detect model drift, data quality issues
- **When**: Production deployment
- **Current**: Manual result inspection (fine for demos)

---

## 10. Summary: Key Principles

1. **Correctness Over Speed**: Algorithms must be correct; speed optimizations come later
2. **Interpretability Over Complexity**: Simple, understandable methods over black boxes
3. **Proven Over Experimental**: Use well-validated methods from research/industry
4. **General Over Specific**: Algorithms that work across datasets and use cases
5. **Documentation Over Code**: Explain reasoning so code is maintainable
6. **Production-Ready Design**: Code structure supports production even if not fully optimized

---

## References

All citations and detailed references are in:
- "demo/CITATIONS.md": Complete bibliography
- "TECHNICAL_REPORT.md": Detailed methodology explanations
- Individual module docstrings: Algorithm-specific reasoning

