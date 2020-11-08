import argparse
import logging
import sys
import tabulate

from kubernetes import client, config
from kubernetes.client import V1EnvVarSource
from kubernetes.config import ConfigException

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

config.load_kube_config()
#configuration = client.Configuration()
#api_instance = client.BatchV1Api(client.ApiClient(configuration))
batch_v1 = client.BatchV1Api()
core_v1 = client.CoreV1Api()


def main():
    pass

def submit_job(args):
    container_image = args.container
    container_name = args.name

    body = client.V1Job(api_version="batch/v1", kind="Job", metadata=client.V1ObjectMeta(name=container_name))
    body.status = client.V1JobStatus()
    template = client.V1PodTemplate()

    labels = {
        'hugin-job': "1",
        'hugin-job-name': f'{container_name}'
    }
    template.template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels)
    )
    env_list = []

    tolerations = []
    containe_args = dict(name=f"container-{container_name}", image=container_image, env=env_list)
    if args.gpu:
        tolerations.append(client.V1Toleration(
        key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        containe_args['resources'] = client.V1ResourceRequirements(limits={"nvidia.com/gpu": 1})

    container = client.V1Container(**containe_args)
    pull_secrets = []
    if args.pull_secret is not None:
        pull_secrets.append(client.V1LocalObjectReference(name=args.pull_secret))
    pod_args = dict(containers=[container],
                    restart_policy='Never',
                    image_pull_secrets=pull_secrets)


    if tolerations:
        pod_args['tolerations'] = tolerations

    if args.node_selector is not None:
        parts = args.node_selector.split("=", 1)
        if len(parts) == 2:
            affinity = client.V1Affinity(
                node_affinity=client.V1NodeAffinity(
                    required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                        node_selector_terms=[client.V1NodeSelectorTerm(
                            match_expressions=[client.V1NodeSelectorRequirement(
                                key=parts[0], operator='In', values=[parts[1]])]
                        )]
                    )
                )
            )
            pod_args['affinity'] = affinity

    template.template.spec = client.V1PodSpec(**pod_args)
    body.spec = client.V1JobSpec(ttl_seconds_after_finished=1800, template=template.template)
    try:
        api_response = batch_v1.create_namespaced_job("default", body, pretty=True)
    except client.exceptions.ApiException as e:
        logging.critical(f"Failed to start job: {e.reason}")

def get_jobs(args):
    result = core_v1.list_namespaced_pod(namespace=args.namespace, label_selector='hugin-job')
    #print (result.items)
    headers = ["Name", "Node Name", "phase"]
    table = []
    for pod in result.items:
        table.append([pod.metadata.name, pod.spec.node_name, pod.status.phase])
    print(tabulate.tabulate(table, headers)) # tablefmt="fancy_grid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HuginEO -- Tool for Kubernetes jobs')
    subparsers = parser.add_subparsers(help='Available commands')

    submit_parser = subparsers.add_parser('submit', help="Submit Jobs")
    submit_parser.add_argument('--name', type=str, required=True, default=None,
                              help='name of the jobs')
    submit_parser.add_argument('--container', type=str, required=True, default=None,
                               help='Id of the container')
    submit_parser.add_argument('--pull-secret', type=str, required=False, default='dev-info-secret', help="The Kubernetes Pull Secret to be used")
    submit_parser.add_argument("--gpu", action='store_true', default=False)
    submit_parser.add_argument("--node-selector", type=str, default="sage-gpu=v100-transient", help="selector=value, eg: sage-gpu=v100-dedicated")
    #submit_parser.add_argument()
    submit_parser.add_argument('--labels', type=str, required=False, default=None,
                               help='Id of the container')
    submit_parser.set_defaults(func=submit_job)
    jobs_parser = subparsers.add_parser('jobs', help="Get Jobs")
    jobs_parser.add_argument("--namespace", type=str, default="default")
    jobs_parser.set_defaults(func=get_jobs)
    get_log_parser = subparsers.add_parser('get-logs', help="Get Logs")
    get_log_parser = subparsers.add_parser('get-logs', help="Cancel Job")

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)

    main()