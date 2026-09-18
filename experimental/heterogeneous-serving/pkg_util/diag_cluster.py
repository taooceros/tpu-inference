import ray

ray.init(address="auto")
print("Cluster resources:", ray.cluster_resources())
print("Available resources:", ray.available_resources())

nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("TPU", 0) > 0]
print(f"Alive TPU nodes ({len(nodes)}):")
for n in nodes:
    ip = n["NodeManagerAddress"]
    res = n["Resources"]
    print(f"Node: {ip}, Resources: {res}")

@ray.remote
def ping_node(ip):
    import os, socket
    return f"Pong from IP {ip}"

futures = {}
for n in nodes:
    ip = n["NodeManagerAddress"]
    futures[ip] = ping_node.options(resources={f"node:{ip}": 1}).remote(ip)

for ip, f in futures.items():
    try:
        res = ray.get(f, timeout=10)
        print(f"Ping success for {ip}: {res}")
    except Exception as e:
        print(f"Ping failed for {ip}: {e}")
