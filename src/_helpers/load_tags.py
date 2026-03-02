from kubernetes import client, config

def load_tags(tags: dict, namespace: str, pod_name: str) -> dict:
    config.load_incluster_config()
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        current_ns = f.read().strip()
    v1 = client.CoreV1Api()
    cm_name = f"{namespace}-aml-details"
    cm = v1.read_namespaced_config_map(name=cm_name, namespace=current_ns)

    for key, value in cm.data.items():
        if value != "undefined":
            if key.lower() == "wbs":
                tags["WBS"] = value
            elif key.lower() == "subproject":
                tags["SubProject"] = value
            else:
                tags[key] = value

    tags.update({
        "submitted_by": namespace,
        "notebook_pod": pod_name,
    })

    return tags
