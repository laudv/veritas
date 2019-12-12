import logging
from dask.distributed import Client, LocalCluster

LOCAL_HOST="localhost"
LOCAL_PORT=30333

def start_local():
    cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            host=LOCAL_HOST,
            scheduler_port=LOCAL_PORT,
            memory_limit="2GB",
            silence_logs=logging.INFO)

    client = Client(cluster.scheduler_address)
    return client

def get_local_client():
    clientname = f"tcp://{LOCAL_HOST}:{LOCAL_PORT}"
    return Client(clientname)

if __name__ == "__main__":
    with start_local() as client:
        print("Dask client:")
        print(client)
        while input("Type 'y' to exit: ") != "y":
            print(client)
