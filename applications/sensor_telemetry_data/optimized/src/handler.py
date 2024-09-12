#from profiler import dump_stats
import json
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

def handler(event, context):
    if 'body' in event:
        event = json.loads(event['body'])
    # Step 1: Load input data
    input_data = event.get('historical_data', [])
    future_periods = event.get('future_periods', 24)


    # Step 2: Prepare the data
    start = datetime(1970, 1, 1)  # Unix epoch start time
    data = pd.DataFrame(input_data)

    # Convert Unix time to datetime
    data['datetime'] = data['ts'].apply(lambda x: start + timedelta(seconds=x))

    # Prepare data for Prophet (ds and y columns)
    df = pd.DataFrame()
    df['ds'] = data['datetime']
    df['y'] = data['smoke']  # Smoke readings

    # Step 3: Initialize and train the model
    m = Prophet()
    m.fit(df)

    # Step 4: Create future dataframe for prediction
    future = m.make_future_dataframe(periods=future_periods, freq='H')

    # Step 5: Make predictions
    forecast = m.predict(future)

    # Step 6: Convert forecast to JSON-like format
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')

    #dump_stats("sensor_telemetry_data_optimized1")
    # Step 7: Return the result as a JSON response
    return {
        'statusCode': 200,
        'body': json.dumps(forecast_result, default=str)  # Convert datetime to string
    }
