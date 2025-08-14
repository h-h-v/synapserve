import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
f = pd.read_csv("ITSM_data.csv")
# Parse datetime using format='mixed' and extract year
f["Open_Time"] = pd.to_datetime(f["Open_Time"], format="mixed", dayfirst=True)
f["Year"] = f["Open_Time"].dt.year

# Count tickets per year
ticket_counts_by_year = f["Year"].value_counts().sort_index()

# Display results
print("\n🎫 Ticket counts by year:")
print(ticket_counts_by_year)
print(len(f))
result2 = f
f = f[f['Open_Time'].notnull()]
print(len(f))
import pandas as pd
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import itertools

# Ensure your datetime column is properly parsed
result2['Open_Time'] = pd.to_datetime(result2['Open_Time'], errors='coerce')
result2 = result2.dropna(subset=['Open_Time'])  # Drop rows with invalid dates

# Extract year from Open_Time
result2['Year'] = result2['Open_Time'].dt.year

# Group by CI_Subcat and Year
yearly_counts = result2.groupby(['CI_Subcat', 'Year']).size().reset_index(name='ticket_count')
categories = yearly_counts['CI_Subcat'].unique()
forecast_years = 3

# Prepare the plot
plt.figure(figsize=(14, 7))

# Style cyclers for multiple lines
colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
linestyles = itertools.cycle(['-', '--', '-.', ':'])

for cat in categories:
    ts_data = yearly_counts[yearly_counts['CI_Subcat'] == cat].sort_values('Year')
    ts_data.set_index('Year', inplace=True)
    ts = ts_data['ticket_count']

    # Skip categories with too few data points
    if len(ts) < 3:
        print(f"⛔ Not enough data for ARIMA forecast: {cat}")
        continue

    try:
        # Fit ARIMA model
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast future values
        forecast = model_fit.forecast(steps=forecast_years)
        last_year = ts.index[-1]
        forecast_index = [last_year + i for i in range(1, forecast_years + 1)]
        forecast_series = pd.Series(forecast.values, index=forecast_index)

        # Assign color and line styles
        color = next(colors)
        linestyle_hist = next(linestyles)
        linestyle_forecast = ':'

        # Plot historical
        plt.plot(ts.index, ts.values, label=f'{cat} - Historical', marker='o', linestyle=linestyle_hist, color=color)

        # Plot forecast
        plt.plot(forecast_series.index, forecast_series.values, label=f'{cat} - Forecast', marker='x', linestyle=linestyle_forecast, color=color)

        # Optional debug:
        # print(forecast_series)

    except Exception as e:
        print(f"⚠️ ARIMA failed for {cat}: {e}")

# Final plot settings
plt.title("Ticket Count Forecast for All CI_Subcat Categories")
plt.xlabel("Year")
plt.ylabel("Ticket Count")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import itertools

forecast_years = 3
categories = yearly_counts['CI_Subcat'].unique()

plt.figure(figsize=(14, 7))

colors = itertools.cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
linestyles = itertools.cycle(['-', '--', '-.', ':'])

for cat in categories:
    ts_data = yearly_counts[yearly_counts['CI_Subcat'] == cat].sort_values('Year').reset_index(drop=True)
    ts_data.set_index('Year', inplace=True)
    ts = ts_data['ticket_count']
    
    color = next(colors)
    linestyle_hist = next(linestyles)
    linestyle_forecast = ':'

    # Plot historical data
    plt.plot(ts.index, ts.values, label=f'{cat} - Historical', marker='o', color=color, linestyle=linestyle_hist)

    # Naive forecast: repeat last observed value for forecast years
    last_value = ts.iloc[-1]
    forecast_index = range(ts.index[-1] + 1, ts.index[-1] + 1 + forecast_years)
    forecast_values = [last_value] * forecast_years
    
    plt.plot(forecast_index, forecast_values, label=f'{cat} - Naive Forecast', marker='x', color=color, linestyle=linestyle_forecast)

plt.title("Ticket Count Naive Forecast for All CI_Subcat Categories")
plt.xlabel('Year')
plt.ylabel('Ticket Count')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True)
plt.tight_layout()
plt.show()
