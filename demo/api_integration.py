"""
Public API Integration for Incrementality Measurement

Demonstrates integration with public APIs that can provide real-world data
for incrementality measurement and analysis.

References:
- Public APIs Repository: https://github.com/public-apis/public-apis
- Various public APIs for advertising, analytics, and financial data
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta


class APIIntegration:
    """
    Integration with public APIs for incrementality measurement.
    
    This module demonstrates how to fetch real-world data from public APIs
    to use with incrementality measurement algorithms.
    """
    
    def __init__(self):
        """Initialize API integration."""
        self.session = requests.Session()
    
    def fetch_exchange_rates(
        self,
        base_currency: str = 'USD',
        target_currencies: List[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Fetch exchange rates for revenue conversion.
        
        Uses ExchangeRate-API or similar public API.
        Useful for normalizing revenue across different currencies.
        
        Args:
            base_currency: Base currency code (default 'USD')
            target_currencies: List of target currency codes
            api_key: Optional API key (if required)
            
        Returns:
            Dictionary with exchange rates
        """
        if target_currencies is None:
            target_currencies = ['EUR', 'GBP', 'JPY']
        
        # Example: ExchangeRate-API (free tier available)
        # URL: https://www.exchangerate-api.com/
        # Note: This is a demonstration - actual implementation would use real API
        
        if not api_key:
            raise ValueError("API key required for exchange rate API. Set api_key parameter.")
        
        rates = {}
        for currency in target_currencies:
            response = self.session.get(
                f'https://api.exchangerate-api.com/v4/latest/{base_currency}',
                params={'api_key': api_key}
            )
            response.raise_for_status()
            data = response.json()
            rates[currency] = data['rates'][currency]
        
        return rates
    
    def fetch_market_data(
        self,
        symbols: List[str],
        date_from: str,
        date_to: str,
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch market data for revenue/campaign analysis.
        
        Uses market data APIs like Alpha Vantage, Yahoo Finance, etc.
        Useful for contextualizing campaign performance with market trends.
        
        Args:
            symbols: List of stock/crypto symbols
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            api_key: Optional API key
            
        Returns:
            DataFrame with market data
        """
        if not api_key:
            raise ValueError("API key required for market data API. Set api_key parameter.")
        
        data = []
        for symbol in symbols:
            response = self.session.get(
                f'https://www.alphavantage.co/query',
                params={
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': api_key
                }
            )
            response.raise_for_status()
            json_data = response.json()
            
            if 'Time Series (Daily)' in json_data:
                ts = json_data['Time Series (Daily)']
                for date, values in ts.items():
                    data.append({
                        'symbol': symbol,
                        'date': date,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })
        
        return pd.DataFrame(data)
    
    def fetch_geographic_data(
        self,
        locations: List[str],
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch geographic data for geo-holdout experiments.
        
        Uses geocoding APIs like OpenStreetMap, Google Maps, etc.
        Useful for geo-experiment design and validation.
        
        Args:
            locations: List of location names or coordinates
            api_key: Optional API key
            
        Returns:
            DataFrame with geographic data
        """
        data = []
        for location in locations:
            response = self.session.get(
                'https://nominatim.openstreetmap.org/search',
                params={
                    'q': location,
                    'format': 'json',
                    'limit': 1
                },
                headers={'User-Agent': 'IncrementalityMeasurement/1.0'}
            )
            response.raise_for_status()
            results = response.json()
            
            if results:
                result = results[0]
                data.append({
                    'location': location,
                    'lat': float(result.get('lat', 0)),
                    'lon': float(result.get('lon', 0)),
                    'display_name': result.get('display_name', ''),
                    'type': result.get('type', '')
                })
        
        return pd.DataFrame(data)
    
    def normalize_revenue_by_currency(
        self,
        revenue_data: pd.DataFrame,
        currency_column: str = 'currency',
        amount_column: str = 'amount',
        target_currency: str = 'USD'
    ) -> pd.DataFrame:
        """
        Normalize revenue data by currency using exchange rates.
        
        Useful for international campaigns where revenue is in different currencies.
        
        Args:
            revenue_data: DataFrame with revenue data
            currency_column: Column name with currency codes
            amount_column: Column name with amounts
            target_currency: Target currency for normalization
            
        Returns:
            DataFrame with normalized revenue
        """
        # Fetch exchange rates
        currencies = revenue_data[currency_column].unique().tolist()
        rates = self.fetch_exchange_rates(target_currency, currencies)
        
        # Normalize to target currency
        revenue_data['amount_usd'] = revenue_data.apply(
            lambda row: row[amount_column] / rates.get(row[currency_column], 1.0),
            axis=1
        )
        
        return revenue_data


def fetch_campaign_metadata(
    api_key: Optional[str] = None
) -> Dict[str, any]:
    """
    Fetch campaign metadata from advertising platforms.
    
    Demonstrates integration with advertising APIs like:
    - Facebook Marketing API
    - Google Ads API
    - Twitter Ads API
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Dictionary with campaign metadata
    """
    # Note: Actual implementation would use real API endpoints
    # Example structure:
    # campaigns = {
    #     'campaign_id': '12345',
    #     'name': 'Campaign Name',
    #     'start_date': '2024-01-01',
    #     'end_date': '2024-01-31',
    #     'budget': 10000.0,
    #     'spend': 9500.0,
    #     'impressions': 100000,
    #     'clicks': 5000,
    #     'conversions': 100,
    #     'revenue': 5000.0
    # }
    
    return {}


def integrate_api_data_with_incrementality(
    revenue_data: pd.DataFrame,
    spend_data: pd.DataFrame,
    test_group: np.ndarray,
    control_group: np.ndarray
) -> Dict[str, any]:
    """
    Integrate API-fetched data with incrementality calculations.
    
    Combines real API data with incrementality measurement algorithms.
    
    Args:
        revenue_data: Revenue data from APIs
        spend_data: Ad spend data from APIs
        test_group: Test group indicators
        control_group: Control group indicators
        
    Returns:
        Dictionary with incrementality metrics
    """
    # Calculate incrementality using API data
    test_revenue = revenue_data[test_group]['amount'].sum()
    control_revenue = revenue_data[control_group]['amount'].sum()
    total_spend = spend_data['amount'].sum()
    
    # Calculate iROAS
    incremental_revenue = test_revenue - control_revenue
    iroas = (incremental_revenue / total_spend * 100.0) if total_spend > 0 else 0.0
    
    return {
        'test_revenue': test_revenue,
        'control_revenue': control_revenue,
        'incremental_revenue': incremental_revenue,
        'total_spend': total_spend,
        'iroas': iroas
    }


def fetch_time_series_data(
    metric: str,
    start_date: str,
    end_date: str,
    granularity: str = 'daily',
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch time series data for causal impact analysis.
    
    Uses time series APIs for:
    - Pre-period/post-period analysis
    - DID (Difference-in-Differences) studies
    - Time-based incrementality measurement
    
    Args:
        metric: Metric name (e.g., 'revenue', 'conversions')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        granularity: Data granularity ('daily', 'weekly', 'monthly')
        api_key: Optional API key
        
    Returns:
        DataFrame with time series data
    """
    # Placeholder for actual API integration
    # In real implementation, would fetch from:
    # - Google Analytics API
    # - Facebook Analytics API
    # - Custom analytics APIs
    
    return pd.DataFrame()


# Example usage and API recommendations
API_RECOMMENDATIONS = {
    'exchange_rates': {
        'apis': [
            'ExchangeRate-API (https://www.exchangerate-api.com/)',
            'CurrencyLayer (https://currencylayer.com/)',
            'Fixer.io (https://fixer.io/)'
        ],
        'use_case': 'Normalize revenue across different currencies for international campaigns'
    },
    'market_data': {
        'apis': [
            'Alpha Vantage (https://www.alphavantage.co/)',
            'Yahoo Finance API',
            'Marketstack (https://marketstack.com/)'
        ],
        'use_case': 'Contextualize campaign performance with market trends'
    },
    'geographic_data': {
        'apis': [
            'OpenStreetMap Nominatim (https://nominatim.openstreetmap.org/)',
            'Google Geocoding API',
            'Mapbox Geocoding API'
        ],
        'use_case': 'Geo-holdout experiment design and validation'
    },
    'advertising_data': {
        'apis': [
            'Facebook Marketing API',
            'Google Ads API',
            'Twitter Ads API',
            'LinkedIn Marketing API'
        ],
        'use_case': 'Fetch real campaign data for incrementality measurement'
    },
    'analytics_data': {
        'apis': [
            'Google Analytics API',
            'Adobe Analytics API',
            'Mixpanel API'
        ],
        'use_case': 'Time series data for causal impact analysis'
    }
}


# Example usage
if __name__ == '__main__':
    print("Public API Integration for Incrementality Measurement")
    print("=" * 80)
    print("\nThis module demonstrates integration with public APIs from:")
    print("https://github.com/public-apis/public-apis")
    print("\nRecommended APIs for incrementality measurement:")
    
    for category, info in API_RECOMMENDATIONS.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Use Case: {info['use_case']}")
        print(f"  APIs:")
        for api in info['apis']:
            print(f"    - {api}")
    
    print("\n" + "=" * 80)
    print("Note: Actual API integration requires API keys and implementation")
    print("of specific API endpoints. This module provides a framework for")
    print("integrating real-world data with incrementality measurement algorithms.")

