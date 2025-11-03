# Data Explanation

Overview of datasets used in the incremental advertising measurement demo.

## Datasets

### 1. Real-time Advertisers Auction (Dataset.csv)
- **Source**: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
- **Size**: ~477 MB, ~567K records (10K sample for demo)
- **Structure**: 
  - 17 columns including:
    - total_revenue: Revenue from conversions
    - total_impressions: Number of impressions served
    - date: Timestamp for temporal analysis
    - Additional feature columns: site_id, geo_id, device_category_id, advertiser_id, etc.
- **Note**: This dataset contains actual revenue data, making it suitable for iROAS calculations.

### 2. Video Ads Engagement (ad_df.csv)
- **Source**: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset
- **Size**: ~437 MB, ~3M records (10K sample for demo)
- **Structure**: 17 columns including "seconds_played" (outcome), "creative_duration" (spend proxy), "timestamp", and user/contextual features

## Data Processing

- **Sample Size**: 10K rows per dataset for demo purposes
- **Treatment Assignment**: Random 50/50 split using binomial distribution (RCT methodology)
- **Outcome**: Binary conversion for Real-time Auction (revenue > 0), continuous engagement for Video Ads
- **Features**: First 10 numeric columns used as input features
- **Spend Calculation**: For Real-time Auction, spend is estimated from impressions using CPM ($2.00 per 1000 impressions)

## Limitations

- **Real-time Auction**: No actual bid prices (spend is estimated from impressions)
- **Real-time Auction**: Ghost bidding demo cannot run (requires bid_price or competition_bid columns)
- **Low conversion rates**: ~0.1% baseline (typical of real advertising)
