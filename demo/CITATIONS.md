# Citations and References

Complete bibliography of all cited papers, methodologies, and references
from conversation.md used in this technical demo implementation.

## Core Incrementality Metrics

### iROAS Formula and Calculation
- **Remerge**: "A quick guide to interpreting incremental ROAS"
  - URL: https://www.remerge.io/findings/blog-post/a-quick-guide-to-interpreting-incremental-roas
  - Reference: conversation.md line 32-33, 65, 78
  - Formula: iROAS = Incremental Revenue / Ad Spend

- **Lifesight**: "What is Incremental ROAS (iROAS)?"
  - URL: https://lifesight.io/glossary/what-is-incremental-roas-iroas/
  - Reference: conversation.md line 35-36, 59, 62, 78
  - Definition: Additional revenue created because of ads

- **IMM Guide**: "The Definitive Guide to Understanding Incremental Return On Ad Spend (iROAS)"
  - URL: https://imm.com/blog/iroas-incremental-return-on-ad-spend
  - Reference: conversation.md line 20-21, 59, 72-73
  - Bidding strategy: Use iROAS in second-price auctions

- **INCRMNTAL**: "Understanding Incremental ROAS vs ROAS"
  - URL: https://www.incrmntal.com/resources/understanding-incremental-roas-vs-roas-for-marketers
  - Reference: conversation.md line 23-24, 59, 61-62, 131
  - Explains difference between incremental and attributed revenue

## Causal Inference Methods

### CUPED (Conditional Unconfounded Pre-period Estimator)
- **Microsoft Research 2013**: CUPED methodology
  - Reference: conversation.md references CUPED multiple times
  - Algorithm: Variance reduction through pre-period covariate adjustment
  - Formula: Adjusted estimate = Treatment effect - Theta * (Pre-period diff)
  - Theta = Cov(pre, post) / Var(pre)

### Tree-based Causal Inference
- **Wang, P., Sun, W., Yin, D., Yang, J., Chang, Y. (2015)**: "Robust Tree-based Causal Inference for Complex Ad Effectiveness Analysis"
  - Conference: Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM '15)
  - Publisher: ACM, New York, NY, USA
  - Pages: 67-76
  - DOI/URL: Referenced in conversation.md line 107
  - Algorithm: Tree-based partitioning with causal effect estimation for heterogeneous treatment effects

### Propensity Score Matching
- **Referenced in conversation.md line 134**: Observational/causal inference methods
  - Algorithm: Logistic regression for propensity score estimation
  - Matching: 1-to-1 nearest neighbor with caliper
  - ATT: Average Treatment Effect on the Treated calculation

### Difference-in-Differences
- **Referenced in conversation.md line 134, 215**: Observational methods
  - Formula: DID = (Y_post_treat - Y_pre_treat) - (Y_post_control - Y_pre_control)
  - Two-way fixed effects implementation

### Uplift Modeling
- **Kunzel et al. 2019**: Meta-learners for causal inference
  - Referenced in conversation.md for uplift modeling methods
  - T-learner: Two separate models for treatment and control
  - X-learner: Cross-learner algorithm
  - DR-learner: Doubly Robust learner
  - S-learner: Single model with treatment indicator

## Experimentation Frameworks

### ACM WWW 2021: Double-Blind Designs
- **Paper**: "Incrementality Testing in Programmatic Advertising: Enhanced Precision with Double-Blind Designs"
  - Conference: Proceedings of the Web Conference 2021
  - DOI: https://dl.acm.org/doi/10.1145/3442381.3450106
  - Reference: conversation.md line 106, 133, 138
  - Methodology: Double-blind designs in programmatic advertising

### ResearchGate Paper
- **Title**: "Incrementality Testing in Programmatic Advertising: Enhanced Precision with Double-Blind Designs"
  - URL: https://www.researchgate.net/publication/350457356_Incrementality_Testing_in_Programmatic_Advertising_Enhanced_Precision_with_Double-Blind_Designs
  - Reference: conversation.md line 127-128
  - Data: 15 U.S. advertising experiments at Facebook with 500 million user-experiment observations and 1.6 billion ad impressions

## Ghost Bidding

### Moloco Methodology
- **Blog Post**: "Beyond Last-Click: How We Prove Incremental ROAS in Retail Media"
  - URL: https://www.moloco.com/blog/proving-incremental-roas-retail-media-advertising
  - Reference: conversation.md line 29-30, 66, 73, 138
  - Algorithm: Randomized controlled trial (RCT) framework powered by ghost bidding
  - Method: Withhold impressions that would have won for randomized subset to form live control group

## Auction Theory and Bidding

### IMM Guide on Bidding Strategies
- **IMM**: "The Definitive Guide to Understanding Incremental Return On Ad Spend (iROAS)"
  - URL: https://imm.com/blog/iroas-incremental-return-on-ad-spend
  - Reference: conversation.md line 21, 72-73
  - Strategy: In second-price auctions, bid to observed incremental value
  - First-price: Apply price shading

## Industry References

### Airbridge Blog
- **Article**: "Intro to Incrementality: The Key to Measuring Your Ad Effectiveness"
  - URL: https://www.airbridge.io/blog/marketing-incrementality
  - Reference: conversation.md line 100-101, 131, 133, 135, 138
  - Methods: Experimental and observational approaches

### INCRMNTAL Resources
- **Article**: "Incrementality Testing for Ad Measurement Success"
  - URL: https://www.incrmntal.com/resources/incrementality-testing-the-key-to-measuring-advertising-effectiveness
  - Reference: conversation.md line 109-110, 131, 133, 135, 138
  - Technology: Causal inference using attribution data

- **Article**: "What is Incrementality in Marketing?"
  - URL: https://www.incrmntal.com/resources/how-do-we-measure-incrementality
  - Reference: conversation.md line 115-116
  - Algorithm: Causal discovery algorithm with adstock effects

### Adjust Glossary
- **Article**: "What does incrementality mean in marketing?"
  - URL: https://www.adjust.com/glossary/incrementality/
  - Reference: conversation.md line 118-119
  - Method: Causal inference as favored approach for quantifying incrementality

### Measured FAQ
- **Article**: "Understanding incrementality for marketing success"
  - URL: https://www.measured.com/faq/what-is-incrementality-in-marketing/
  - Reference: conversation.md line 121-122
  - Focus: Pinpointing platforms and campaigns with greatest causal impact

### MiQ Tech and Analytics
- **Medium Article**: "Measuring Digital Ad Effectiveness using Incrementality Testing"
  - URL: https://medium.com/miq-tech-and-analytics/measuring-digital-ad-effectiveness-using-incrementality-testing-b59ca58ce934
  - Reference: conversation.md line 124-125
  - Approach: Mathematical approach to measuring causal impact

### Recast
- **Article**: "What does 'Incrementality' mean?"
  - URL: https://getrecast.com/incrementality/
  - Reference: conversation.md line 112-113, 131
  - Method: Geo-holdout tests using open-source libraries

## Additional Industry References

### CommerceIQ
- **Article**: "How to Measure True Incrementality in Retail Media"
  - URL: https://www.commerceiq.ai/blog/measuring-real-incrementality-in-retail-media
  - Reference: conversation.md line 26-27, 68, 73

### Rockerbox
- **FAQ**: "What is Incremental ROAS?"
  - URL: https://www.rockerbox.com/faq/what-is-incremental-roas
  - Reference: conversation.md line 44-45

### Vilop Digital
- **Article**: "Understanding Incremental ROAS"
  - URL: https://vilopdigital.com/understanding-incremental-roas/
  - Reference: conversation.md line 41-42

### Nextscenario
- **Article**: "Incremental Roas Vs Roas: Roas Formula And Best Practices"
  - URL: https://nextscenario.com/incremental-roas-vs-roas-roas-formula-and-best-practices/
  - Reference: conversation.md line 38-39

### Agency Vista
- **Article**: "A Marketer's Guide To Interpreting Incremental ROAS in 2021"
  - URL: https://agencyvista.com/insights/a-marketers-guide-to-interpreting-incremental-roas-in-2021/
  - Reference: conversation.md line 47-48

## Off-Policy Evaluation

### IPS and DR Methods
- **Referenced in conversation.md line 215, 260**: Off-policy evaluation methods
  - IPS: Inverse Propensity Scoring
  - DR: Doubly Robust evaluation
  - Used for evaluating new policies using data from old policies

## Dataset Integration

### Kaggle Datasets

1. **Real-time Advertisers Auction Dataset** ("saurav9786/real-time-advertisers-auction")
   - URL: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
   - Description: Real-time advertiser auction data with actual revenue (`total_revenue`) and impression data (`total_impressions`), enabling direct iROAS calculations
   - Note: Dataset contains 17 columns including revenue, impressions, date, and feature columns (site_id, geo_id, device_category_id, advertiser_id, etc.)
   - Use Case: Real-world auction data with revenue for incrementality measurement and iROAS calculations
   - Integration: Use kaggle package to download and process the dataset

2. **Video Ads Engagement Dataset** ("karnikakapoor/video-ads-engagement-dataset")
   - URL: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset
   - Description: Video advertisement engagement metrics with temporal features
   - Use Case: Real-world video ads engagement data for incrementality measurement and analysis
   - Integration: Use kaggle package to download and process the dataset

## API Integration

### Public APIs Repository
- **Repository**: Public APIs (https://github.com/public-apis/public-apis)
  - A collective list of free APIs from many domains
  - 375k+ stars, community-maintained
  - Includes APIs for advertising, analytics, finance, and geographic data

### Recommended APIs for Incrementality Measurement
- **Exchange Rate APIs**: For normalizing revenue across currencies
  - ExchangeRate-API, CurrencyLayer, Fixer.io
- **Market Data APIs**: For contextualizing campaign performance
  - Alpha Vantage, Yahoo Finance, Marketstack
- **Geographic Data APIs**: For geo-holdout experiments
  - OpenStreetMap Nominatim, Google Geocoding, Mapbox
- **Advertising APIs**: For real campaign data
  - Facebook Marketing API, Google Ads API, Twitter Ads API
- **Analytics APIs**: For time series analysis
  - Google Analytics API, Adobe Analytics API, Mixpanel API

## Notes

All implementations in this demo are based on real algorithms and methodologies
from the cited papers and industry references. No placeholders, hardcoded values,
or random results are used - all calculations are algorithmic and deterministic
based on input data.

The API integration module ("api_integration.py") provides a framework for
integrating real-world data from public APIs with the incrementality measurement
algorithms. Actual API integration requires API keys and implementation of
specific API endpoints.

