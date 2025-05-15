from datetime import datetime, timedelta
from airflow import DAG
import pandas as pd
from airflow.operators.python import PythonOperator
from train_taxi_model import train_taxi_model

def download_taxi_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Ranbof/Data-lab-3/refs/heads/main/taxi_dataset.csv')
    #df.to_csv("taxi_data.csv", index=False)
    print(f"Downloaded data shape: {df.shape}")
    return True

def preprocess_taxi_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Ranbof/Data-lab-3/refs/heads/main/taxi_dataset.csv")
    
    # Basic cleaning
    df = df.dropna()
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    df = df[(df['total_amount'] > 0) & (df['total_amount'] < 200)]
    df = df[df['passenger_count'] > 0]
    
    # Feature engineering
    df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['pickup_day'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek
    df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
    
    
    df.to_csv('processed_taxi_data.csv', index=False)
    return True

dag_taxi = DAG(
    dag_id="taxi_fare_pipeline",
    start_date=datetime(2025, 2, 3),
    schedule_interval="@daily",
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }
)

download_task = PythonOperator(
    task_id="download_taxi_data",
    python_callable=download_taxi_data,
    dag=dag_taxi
)

preprocess_task = PythonOperator(
    task_id="preprocess_taxi_data",
    python_callable=preprocess_taxi_data,
    dag=dag_taxi
)

train_task = PythonOperator(
    task_id="train_taxi_model",
    python_callable=train_taxi_model,
    dag=dag_taxi
)

download_task >> preprocess_task >> train_task