# Orchest
import logging
from azure.durable_functions import DurableOrchestrationContext, Orchestrator
import azure.durable_functions as df

logging.basicConfig(level=logging.INFO)


def orchestrator_function(context: df.DurableOrchestrationContext):
    try:
        input_data = context.get_input()

        if not input_data:
            logging.error("No input data received.")
            return {"error": "No input data received."}

        region_cur_name = input_data.get("region_cur_name")
        file_path = input_data.get("file_path")

        if not region_cur_name or not file_path:
            logging.error("Missing required input parameters.")
            return {"error": "Missing required input parameters."}

        logging.info(
            f"Processing data for region: {region_cur_name} with file: {file_path}")

        # Process Transformation
        transformation_result = yield context.call_activity("Transformation", {
            "region_cur_name": region_cur_name,
            "file_path": file_path
        })

        # Extract processed_data from transformation result
        processed_data = transformation_result.get("processed_data")
        if not processed_data:
            raise ValueError("No processed data received from transformation")

        # Save as parquet file
        parquet_result = yield context.call_activity("JsonToParquet", {
            "data": processed_data,
            "region": region_cur_name
        })

        return parquet_result

    except Exception as e:
        logging.error(f"Error in orchestrator function: {str(e)}")
        return {"error": str(e)}


main = Orchestrator.create(orchestrator_function)
