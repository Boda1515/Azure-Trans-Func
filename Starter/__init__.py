# HttpStarter
import logging

from azure.functions import HttpRequest, HttpResponse
from azure.durable_functions import DurableOrchestrationClient


async def main(req: HttpRequest, starter: str) -> HttpResponse:
    """
    This function takes two parameters: 'file_path' and 'region'
    It performs a transformation on the 'price' field within the specified region.
    """

    # Create a DurableOrchestrationClient
    client = DurableOrchestrationClient(starter)
    try:
        req_data = req.get_json()
    except:
        req_data = dict(req.params)

    region_cur_name = req_data.get("region_cur_name")
    file_path = req_data.get("file_path")

    if not file_path or not region_cur_name:
        return HttpResponse(
            "Please pass both file_path and region in the request body",
            status_code=400
        )

    # Start the orchestration
    instance_id = await client.start_new("Orchest", None, {
        "region_cur_name": region_cur_name,
        "file_path": file_path
    })

    logging.info(f"Started orchestration with ID = '{instance_id}'.")

    return client.create_check_status_response(req, instance_id)
