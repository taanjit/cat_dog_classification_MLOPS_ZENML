# run the pipeline
python3 run_pipeline.py

# run the predict api
uvicorn app.main:app --reload

# model version history 
mlflow ui --backend-store-uri ./mlruns

# run pipeline monitoring
python3 run_pipeline.py