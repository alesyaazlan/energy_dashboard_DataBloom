import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df_1 = pd.read_csv('smart_home_energy_usage_dataset.csv')

df_1['timestamp'] = pd.to_datetime(df_1['timestamp'])
df_1.rename(columns={'energy_consumption_kWh' : 'energy_consumption'}, inplace=True)


df_2 = pd.read_csv('smart_home_energy_consumption_large.csv')

df_2['timestamp'] = pd.to_datetime(df_2['Date'] + ' ' + df_2['Time'])
df_2.rename(columns={
    'Season' : 'season',
    'Appliance Type' : 'appliance',
    'Energy Consumption (kWh)' : 'energy_consumption'
}, inplace=True)
df_2.replace({
    'season': {
        'Fall' : 'Autumn'
    },
    'appliance':{
        'Lights': 'Lighting',
        'Fridge': 'Refrigerator'
    }
}, inplace=True)


df_1f= df_1[['timestamp', 'energy_consumption', 'season', 'appliance']]
df_2f= df_2[['timestamp', 'energy_consumption', 'season', 'appliance']]


df = pd.concat([df_1f, df_2f])

df = df.groupby(['timestamp', 'season', 'appliance']).mean().reset_index()
df = df[(df['timestamp'] < '2024-01-01')]
df = df.dropna()

df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)




# Set page configuration
st.set_page_config(
    page_title="Home Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)




#region sidebar
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Choose a Page', ['Dashboard', 'Anomaly Detection'])


#region page1
def page1():
    st.title("Home Energy Consumption Dashboard")
    st.markdown("Analyze and visualize energy consumption patterns across different appliances and seasons")


    st.markdown("---")

     # Main Dashboard
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    col1, col2, col3, col4 = st.columns(4)

    total_consumption = df['energy_consumption'].sum()
    avg_daily_consumption = df.groupby(df['timestamp'].dt.date)['energy_consumption'].sum().mean()
    most_consuming_appliance = df.groupby('appliance')['energy_consumption'].sum().idxmax()
    seasonal_consumption = df.groupby('season')['energy_consumption'].sum()
    highest_season = seasonal_consumption.idxmax()

    with col1:
        st.metric(
            "Total Energy Consumption",
            f"{total_consumption:,.0f} kWh",
            help="Sum of all energy consumption in the selected period"
        )

    with col2:
        st.metric(
            "Average Daily Consumption",
            f"{avg_daily_consumption:.1f} kWh/day",
            help="Average daily energy consumption"
        )

    with col3:
        st.metric(
            "Highest Consuming Appliance",
            most_consuming_appliance,
            help="Appliance with the highest total energy consumption"
        )

    with col4:
        st.metric(
            "Highest Consumption Season",
            highest_season,
            help="Season with the highest energy consumption"
        )

    # Charts
    st.markdown("---")

    # Row 1: Time series and seasonal analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy Consumption Over Time")
        
        # Aggregate by date
        daily_consumption = df.groupby(df['timestamp'].dt.date)['energy_consumption'].sum().reset_index()
        
        fig_time = px.line(
            daily_consumption,
            x='timestamp',
            y='energy_consumption',
            title="Daily Energy Consumption Trend",
            labels={'timestamp': 'Date', 'energy_consumption': 'Energy Consumption (kWh)'}
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        st.subheader("Consumption by Season")
        
        seasonal_data = df.groupby('season')['energy_consumption'].sum().reset_index()
        
        fig_season = px.pie(
            seasonal_data,
            values='energy_consumption',
            names='season',
            title="Energy Consumption Distribution by Season",
            hole=0.4
        )
        fig_season.update_layout(height=400)
        st.plotly_chart(fig_season, use_container_width=True)


    # Row 2: Appliance analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy Consumption by Appliance")
        
        appliance_data = df.groupby('appliance')['energy_consumption'].sum().reset_index()
        appliance_data = appliance_data.sort_values('energy_consumption', ascending=True)
        
        fig_appliance = px.bar(
            appliance_data,
            y='appliance',
            x='energy_consumption',
            title="Total Energy Consumption by Appliance",
            labels={'energy_consumption': 'Total Energy Consumption (kWh)', 'appliance': 'Appliance'},
            orientation='h'
        )
        fig_appliance.update_layout(height=400)
        st.plotly_chart(fig_appliance, use_container_width=True)

    with col2:
        st.subheader("Average Daily Consumption by Appliance and Season")
        
        # Calculate average daily consumption by appliance and season
        avg_consumption = df.groupby(['appliance', 'season'])['energy_consumption'].mean().reset_index()
        
        fig_heatmap = px.density_heatmap(
            avg_consumption,
            x='season',
            y='appliance',
            z='energy_consumption',
            title="Average Daily Consumption Heatmap (kWh)",
            color_continuous_scale='Viridis'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Row 3: Monthly trends and comparison
    st.subheader("Monthly Energy Consumption Trends")

    # Extract month and year
    df['month'] = df['timestamp'].dt.month_name()
    df['year'] = df['timestamp'].dt.year

    monthly_data = df.groupby(['year', 'month', 'appliance'])['energy_consumption'].sum().reset_index()

    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['month'] = pd.Categorical(monthly_data['month'], categories=month_order, ordered=True)
    monthly_data = monthly_data.sort_values(['year', 'month'])

    fig_monthly = px.line(
        monthly_data,
        x='month',
        y='energy_consumption',
        color='appliance',
        title="Monthly Energy Consumption by Appliance",
        labels={'energy_consumption': 'Energy Consumption (kWh)', 'month': 'Month'}
    )
    fig_monthly.update_layout(height=500)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Data table
    st.markdown("---")
    st.subheader("Raw Data")

    with st.expander("View Raw Data"):
        st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)
        

#region modelling
def create_forecast_model(df, appliance, model_type='linear'):
    """Create forecasting model and calculate error metrics"""
    appliance_data = df[df['appliance'] == appliance].copy()
    
    if len(appliance_data) < 10:
        return None, None, None, None
    
    # Prepare features
    features = appliance_data[['day_of_week', 'month', 'is_weekend']].copy()
    features['day_of_year'] = appliance_data['timestamp'].dt.dayofyear
    features['rolling_avg_7'] = appliance_data['energy_consumption'].rolling(7, min_periods=1).mean()
    
    target = appliance_data['energy_consumption']
    
    # Remove rows with NaN
    valid_idx = features.notna().all(axis=1)
    features = features[valid_idx]
    target = target[valid_idx]
    
    if len(features) < 10:
        return None, None, None, None
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(features))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    if model_type == 'linear':
        model = LinearRegression()
    else:  # random_forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return model, y_test, y_pred, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def detect_forecast_anomalies(df, appliance, error_threshold=2.0):
    """Detect anomalies based on forecasting errors"""
    model, y_test, y_pred, errors = create_forecast_model(df, appliance, 'random_forest')
    
    if model is None:
        return pd.Series([False] * len(df), index=df.index), None, None, None
    
    # Prepare full dataset for prediction
    appliance_data = df[df['appliance'] == appliance].copy()
    features = appliance_data[['day_of_week', 'month', 'is_weekend']].copy()
    features['day_of_year'] = appliance_data['timestamp'].dt.dayofyear
    features['rolling_avg_7'] = appliance_data['energy_consumption'].rolling(7, min_periods=1).mean()
    
    # Predict for all data
    full_predictions = model.predict(features.fillna(method='bfill'))
    
    # Calculate residuals
    residuals = appliance_data['energy_consumption'] - full_predictions
    residual_std = residuals.std()
    
    # Detect anomalies (points where residual > threshold * std)
    anomalies = np.abs(residuals) > error_threshold * residual_std
    
    # Create result series for full dataframe
    result = pd.Series([False] * len(df), index=df.index)
    result.loc[appliance_data.index] = anomalies
    
    return result, full_predictions, errors, residuals

#region page2
def page2():
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_consumption = df['energy_consumption'].sum()
    avg_consumption = df['energy_consumption'].mean()
    
    with col1:
        st.metric("Total Energy Consumption", f"{total_consumption:,.0f} kWh")
    
    with col2:
        st.metric("Average Consumption", f"{avg_consumption:.2f} kWh")
    
    with col3:
        if 'detected_anomaly' in df.columns:
            anomaly_count = df['detected_anomaly'].sum()
            st.metric("Detected Anomalies", f"{anomaly_count}", 
                    delta=f"{(anomaly_count/len(df)*100):.1f}% of total")
        else:
            st.metric("Detected Anomalies", "N/A")
    
    with col4:
        if 'detected_anomaly' in df.columns and df['detected_anomaly'].sum() > 0:
            anomaly_consumption = df[df['detected_anomaly']]['energy_consumption'].sum()
            percent_anomaly = (anomaly_consumption / total_consumption) * 100
            st.metric("Anomaly Impact", f"{anomaly_consumption:,.0f} kWh",
                    delta=f"{percent_anomaly:.1f}% of total")
        else:
            st.metric("Anomaly Impact", "N/A")
    
    # Anomaly Visualization
    st.markdown("---")
    st.subheader("Anomaly Detection Results")
    
    if 'detected_anomaly' not in df.columns:
        st.info("Configure and run anomaly detection in the sidebar to see results")
    else:
        # Anomaly Summary
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomalies by appliance
            anomaly_by_season = df[df['detected_anomaly']].groupby(df['timestamp'].dt.month).size()
       
            fig_anomaly_appliance = px.bar(
                anomaly_by_season,
                title="Anomalies Detected by Month",
                labels={'value': 'Number of Anomalies', 'month': 'Month'}
            )
            st.plotly_chart(fig_anomaly_appliance, use_container_width=True)
        
        with col2:
            # Anomaly types (high vs low consumption)
            if len(df[df['detected_anomaly']]) > 0:
                df_anomalies = df[df['detected_anomaly']].copy()
                df_anomalies['anomaly_type'] = df_anomalies['energy_consumption'].apply(
                    lambda x: 'High Consumption' if x > avg_consumption else 'Low Consumption'
                )
                
                anomaly_types = df_anomalies['anomaly_type'].value_counts()
                fig_anomaly_types = px.pie(
                    values=anomaly_types.values,
                    names=anomaly_types.index,
                    title="Distribution of Anomaly Types"
                )
                st.plotly_chart(fig_anomaly_types, use_container_width=True)
        
        # Time Series with Anomalies
        st.subheader("Energy Consumption Timeline with Detected Anomalies")
        
        # Aggregate to daily level for cleaner visualization
        daily_data = df.groupby(df['timestamp'].dt.date).agg({
            'energy_consumption': 'sum',
            'detected_anomaly': 'max'  # Mark day as anomalous if any point is anomalous
        }).reset_index()
        
        fig_timeline = go.Figure()
        
        # Normal consumption
        normal_days = daily_data[~daily_data['detected_anomaly']]
        fig_timeline.add_trace(go.Scatter(
            x=normal_days['timestamp'],
            y=normal_days['energy_consumption'],
            mode='lines+markers',
            name='Normal Consumption',
            line=dict(color='blue')
        ))
        
        # Anomalous consumption
        anomaly_days = daily_data[daily_data['detected_anomaly']]
        if len(anomaly_days) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=anomaly_days['timestamp'],
                y=anomaly_days['energy_consumption'],
                mode='markers',
                name='Anomaly Detected',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig_timeline.update_layout(
            title="Daily Energy Consumption with Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Forecasting Visualization
        if 'forecast_data' in st.session_state:
            st.subheader("🔮 Forecasting Results")
            
            forecast_data = st.session_state.forecast_data
            appliance_data = df[df['appliance'] == forecast_data['appliance']].copy()
            
            if forecast_data['predictions'] is not None:
                # Actual vs Predicted
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=appliance_data['timestamp'],
                    y=appliance_data['energy_consumption'],
                    mode='lines',
                    name='Actual Consumption',
                    line=dict(color='blue')
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=appliance_data['timestamp'],
                    y=forecast_data['predictions'],
                    mode='lines',
                    name='Predicted Consumption',
                    line=dict(color='red', dash='dash')
                ))
                
                # Highlight anomalies
                anomaly_data = appliance_data[appliance_data['detected_anomaly']]
                if len(anomaly_data) > 0:
                    fig_forecast.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['energy_consumption'],
                        mode='markers',
                        name='Detected Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig_forecast.update_layout(
                    title=f"Actual vs Predicted Consumption - {forecast_data['appliance']}",
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption (kWh)",
                    height=500
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Anomaly Details Table
        st.subheader("Detailed Anomaly Report")
        
        if len(df[df['detected_anomaly']]) > 0:
            anomalies_df = df[df['detected_anomaly']].sort_values('energy_consumption', ascending=False)
            
            # Calculate how abnormal each point is
            appliance_means = df.groupby('appliance')['energy_consumption'].mean()
            appliance_stds = df.groupby('appliance')['energy_consumption'].std()
            
            anomalies_df['deviation'] = anomalies_df.apply(
                lambda row: (row['energy_consumption'] - appliance_means[row['appliance']]) / appliance_stds[row['appliance']],
                axis=1
            )
            
            # Display key columns
            display_cols = ['timestamp', 'appliance', 'season', 'energy_consumption', 'deviation']
            anomalies_display = anomalies_df[display_cols].copy()
            anomalies_display['deviation'] = anomalies_display['deviation'].round(2)
            anomalies_display['energy_consumption'] = anomalies_display['energy_consumption'].round(2)
            
            st.dataframe(anomalies_display, use_container_width=True)
            
            # Download anomalies
            csv = anomalies_display.to_csv(index=False)
            st.download_button(
                label="Download Anomaly Report",
                data=csv,
                file_name="energy_anomalies_report.csv",
                mime="text/csv"
            )

            # MAE/RMSE Error Metrics Display
            if 'forecast_data' in st.session_state:
                st.markdown("---")
                st.subheader("📈 Forecasting Error Metrics")
                
                forecast_data = st.session_state.forecast_data
                errors = forecast_data['errors']
                
                if errors:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MAE (Mean Absolute Error)", f"{errors['MAE']:.3f} kWh",
                                help="Average absolute difference between predicted and actual values")
                    
                    with col2:
                        st.metric("RMSE (Root Mean Square Error)", f"{errors['RMSE']:.3f} kWh",
                                help="Square root of average squared differences - penalizes large errors more")
                    
                    with col3:
                        st.metric("MAPE (Mean Absolute Percentage Error)", f"{errors['MAPE']:.1f}%",
                                help="Average percentage difference between predicted and actual values")
                    
                    with col4:
                        ratio = errors['RMSE']/errors['MAE'] if errors['MAE'] > 0 else 0
                        st.metric("RMSE/MAE Ratio", f"{ratio:.3f}",
                                help="Ratio > 1 indicates presence of large errors")
                    
                    # Error Interpretation
                    st.info(f"""
                    **Error Metrics Interpretation for {forecast_data['appliance']}:**
                    - **MAE ({errors['MAE']:.3f} kWh):** On average, predictions are off by {errors['MAE']:.3f} kWh
                    - **RMSE ({errors['RMSE']:.3f} kWh):** Larger errors are {ratio:.1f}x more significant than average errors
                    - **MAPE ({errors['MAPE']:.1f}%):** Average prediction error is {errors['MAPE']:.1f}% of actual consumption
                    """)
        else:
            st.info("No anomalies detected with current settings")
        


if page ==  'Dashboard':
    page1()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

    # Appliance filter
    appliances = st.sidebar.multiselect(
        "Select Appliances",
        options=df['appliance'].unique(),
        default=df['appliance'].unique()
    )
    df = df[df['appliance'].isin(appliances)]

    # Season filter
    seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=df['season'].unique(),
        default=df['season'].unique()
    )
    df = df[df['season'].isin(seasons)]

if page ==  'Anomaly Detection':
    st.title("Anomaly Detection")
    st.markdown("Detect abnormal usage behavior")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Anomaly Detection Settings")
        
       
    appliance_for_detection = st.sidebar.selectbox(
        "Analyze Specific Appliance",
        options=df['appliance'].unique()
    )

    error_threshold = st.sidebar.slider(
        "Error Threshold (σ)",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Number of standard deviations from prediction to consider as anomaly"
    )
    
    # Apply anomaly detection
    if st.sidebar.button("Run Anomaly Detection"):
        with st.spinner("Detecting anomalies..."):
            anomalies, predictions, errors, residuals = detect_forecast_anomalies(
                df, appliance_for_detection, error_threshold
            )
            # Store forecast data for display
            st.session_state.forecast_data = {
                'predictions': predictions,
                'errors': errors,
                'residuals': residuals,
                'appliance': appliance_for_detection
            }
            df['detected_anomaly'] = anomalies
            st.success(f"Detected {anomalies.sum()} anomalies!")
    
    page2()