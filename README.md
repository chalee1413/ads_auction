# Incremental Advertising Measurement

**TLDR**: Implementation of six incrementality measurement algorithms using formulas from peer-reviewed papers. Processes production-scale Kaggle datasets (~3M impressions) and demonstrates the gap between attributed revenue (ROAS) and incremental revenue (iROAS). All algorithms verified against published research.

## What This Is

Six methodologies for measuring incremental advertising effectiveness:

1. **CUPED** - Variance reduction for A/B tests (Microsoft Research)
2. **Tree-based Causal Inference** - Heterogeneous treatment effects (Wang et al. 2015)
3. **Uplift Modeling** - Individual treatment effects (Kunzel et al. 2019)
4. **Ghost Bidding** - Control group creation (Moloco methodology)
5. **iROAS vs ROAS** - Incremental vs attributed measurement comparison
6. **Experiment Workflow** - A/B testing framework

Algorithms use formulas from peer-reviewed papers. Tested on production-scale Kaggle datasets (3M+ impressions).

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt

# Run demo
python3.11 -m demo.main
```

Results saved to "demo/results/results.txt".

## Documentation

- **DATA_EXPLANATION.md** - Dataset structure and characteristics
- **ALGORITHM_RATIONALE.md** - Algorithm rationale and citations
- **RESULTS_EXPLANATION.md** - How to interpret results
- **QUALITY_ASSESSMENT.md** - Code and algorithm verification
- **TECHNICAL_REPORT.md** - Methodology and findings
- **DESIGN_DECISIONS.md** - Algorithm selection rationale

## Key Finding

**ROAS vs iROAS Gap**: Analysis shows 1,615 percentage point gap (ROAS: 1,654% vs iROAS: 39%). Only $279 of $11,729 test revenue is incremental. Optimizing on ROAS misallocates budget.

## Project Structure

```
├── demo/                    # Algorithm implementations
│   ├── main.py             # Run all demos
│   ├── results/            # Output files
│   └── *.py                # Algorithm modules
├── data/kaggle/            # Dataset storage
└── *.md                    # Documentation
```

## Research Foundation

Algorithms implement formulas from:
- **Wang et al. (2015)** - Causal trees for ad effectiveness
- **Deng et al. (2013)** - CUPED variance reduction
- **Kunzel et al. (2019)** - Meta-learners for uplift
- **Kennedy (2020)** - Doubly robust methods

See "demo/CITATIONS.md" for complete bibliography.
