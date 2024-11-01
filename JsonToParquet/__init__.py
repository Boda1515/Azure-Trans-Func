import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO, StringIO
from datetime import datetime
import os
import logging
import json
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)


def main(context) -> str:
    try:
        # Extract data from the input context
        if not isinstance(context, dict) or 'data' not in context:
            raise ValueError("Invalid input format")

        processed_data = context['data']
        region = context['region']

        # Convert JSON string to DataFrame using StringIO
        if isinstance(processed_data, str):
            json_data = StringIO(processed_data)
        else:
            json_data = StringIO(json.dumps(processed_data))

        df = pd.read_json(json_data, orient='records')
        row_count = len(df)
        logging.info(f"Successfully loaded DataFrame with {len(df)} rows")

        # Save DataFrame to Parquet in memory
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, engine='pyarrow', index=False)

        # Get current date
        current_date = datetime.now().strftime("%Y%m%d")

        container_name = "depiproject"
        folder_name = "Operation"
        # File name with row count and date
        blob_name = f"{folder_name}/{region}_{row_count}_rows_{current_date}.parquet"

        # Initialize Blob Service Client using Managed Identity
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url="https://depicomparsionstorage.blob.core.windows.net/",
            credential=credential
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name)

        # Upload the Parquet file to Azure Data Lake
        parquet_buffer.seek(0)
        blob_client.upload_blob(parquet_buffer, overwrite=True)

        logging.info("Parquet file uploaded successfully.")
        return {"status": "success", "message": "Parquet file uploaded successfully."}

    except Exception as e:
        error_message = f"Error uploading Parquet file: {str(e)}"
        logging.error(error_message)
        return {"status": "error", "message": error_message}
