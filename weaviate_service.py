import json
import asyncio
import os
import weaviate
import weaviate.classes as wvc
import aiohttp
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateService:
    def __init__(self, collection_name: str, data_url: str, api_key_env_var: str = "COHERE_API_KEY"):
        self.collection_name = collection_name
        self.data_url = data_url
        self.api_key = os.getenv(api_key_env_var)
        self.headers = self._prepare_headers()
        logger.info("Initialized WeaviateService.")

    def _prepare_headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["X-Cohere-Api-Key"] = self.api_key
            logger.info("Cohere API Key is set.")
        else:
            logger.warning("COHERE_API_KEY is not set. Proceeding without it.")
        return headers

    @asynccontextmanager
    async def managed_clients(self):
        """Context manager to handle both Weaviate and HTTP clients"""
        weaviate_client = None
        http_session = None
        try:
            # Initialize clients
            weaviate_client = weaviate.use_async_with_local(headers=self.headers)
            http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))
            async with weaviate_client as client, http_session as session:
                yield client, session
        finally:
            try:
                # Ensure both clients are properly closed
                if http_session and not http_session.closed:
                    await http_session.close()
                    logger.info("Closed HTTP session.")
                if weaviate_client:
                    await weaviate_client.close()
                    logger.info("Closed Weaviate client connection.")
                # Wait a bit to ensure all connections are properly closed
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error during client cleanup: {e}")

    async def fetch_data(self, session: aiohttp.ClientSession) -> list:
        """Asynchronously fetches JSON data from the specified URL."""
        async with session.get(self.data_url) as resp:
            resp.raise_for_status()
            resp_text = await resp.text()
            data = json.loads(resp_text)
            logger.info(f"Fetched and loaded {len(data)} records.")
            return data

    def remove_duplicates(self, data: list, unique_field: str = "Question") -> list:
        unique_entries = set()
        unique_data = []
        for record in data:
            unique_value = record.get(unique_field)
            if unique_value and unique_value not in unique_entries:
                unique_entries.add(unique_value)
                unique_data.append(record)
        logger.info(f"After removing duplicates, {len(unique_data)} unique records remain.")
        return unique_data

    def prepare_data_objects(self, data: list) -> list:
        data_objects = []
        for record in data:
            try:
                properties = {
                    "answer": record["Answer"],
                    "question": record["Question"],
                    "category": record["Category"],
                }
                vector = record.get("vector")
                data_object = wvc.data.DataObject(properties=properties, vector=vector)
                data_objects.append(data_object)
            except KeyError as e:
                logger.warning(f"Missing field {e} in data entry: {record}")
                continue
        logger.info(f"Prepared {len(data_objects)} data objects for insertion.")
        return data_objects

    async def run(self, query_term: str = "animal", query_limit: int = 2):
        """Execute the full workflow with proper resource management."""
        try:
            async with self.managed_clients() as (client, session):
                # Access collection
                collection = client.collections.get(self.collection_name)
                logger.info(f"Accessed '{self.collection_name}' collection.")

                # Fetch and process data
                data = await self.fetch_data(session)
                unique_data = self.remove_duplicates(data)
                data_objects = self.prepare_data_objects(unique_data)

                # Insert data
                logger.info(f"Inserting {len(data_objects)} objects into the collection '{self.collection_name}'.")
                await collection.data.insert_many(data_objects)
                logger.info("Data insertion completed.")

                # Perform BM25 query
                logger.info(f"Performing BM25 query with '{query_term}'.")
                query_results = await collection.query.bm25(query=query_term, limit=query_limit)
                logger.info("BM25 query completed.")

                # Handle results
                if query_results and query_results.objects:
                    for obj in query_results.objects:
                        print(obj.properties)
                else:
                    logger.warning("No objects returned from the query.")
                logger.info("Processing completed.")

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise

        finally:
            try:
                await asyncio.sleep(0.1)  # Ensure any pending cleanup completes
                logger.info("Cleanup completed.")
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")

async def main():
    collection_name = "JeopardyQuestion"
    data_filename = "jeopardy_tiny_with_vectors_all-OpenAI-ada-002.json"
    data_url = f"https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/{data_filename}"
    query_term = "rattlesnake"
    query_limit = 2

    service = WeaviateService(collection_name=collection_name, data_url=data_url)
    try:
        await service.run(query_term=query_term, query_limit=query_limit)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        await asyncio.sleep(0.1)  # Give time for connections to close

if __name__ == "__main__":
    asyncio.run(main())