import argparse
import logging
import sys

from kubernetes import client, config
from kubernetes.client import V1EnvVarSource
from kubernetes.config import ConfigException

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

config.load_kube_config()
#configuration = client.Configuration()
#api_instance = client.BatchV1Api(client.ApiClient(configuration))
batch_v1 = client.BatchV1Api()


def main():
    pass

def submit_job(args):
    container_image = args.container
    container_name = args.name

    labels = {
        'hugin-job': "1",
        'hugin-job-name': f'{container_name}'
    }

    body = client.V1Job(api_version="batch/v1", kind="Job", metadata=client.V1ObjectMeta(name=container_name))
    body.status = client.V1JobStatus()
    template = client.V1PodTemplate()
    template.template = client.V1PodTemplateSpec(metadata=client.V1ObjectMeta(labels=labels))
    env_list = []
    container = client.V1Container(name=f"containter-{container_name}", image=container_image, env=env_list)
    template.template.spec = client.V1PodSpec(containers=[container], restart_policy='Never', image_pull_secrets=[client.V1LocalObjectReference(name='dev-info-secret')])
    body.spec = client.V1JobSpec(ttl_seconds_after_finished=600, template=template.template)
    api_response = batch_v1.create_namespaced_job("default", body, pretty=True)
    print (api_response)

def get_jobs(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HuginEO -- Tool for Kubernetes jobs')
    subparsers = parser.add_subparsers(help='Available commands')

    submit_parser = subparsers.add_parser('submit', help="Submit Jobs")
    submit_parser.add_argument('--name', type=str, required=True, default=None,
                              help='name of the jobs')
    submit_parser.add_argument('--container', type=str, required=True, default=None,
                               help='Id of the container')
    submit_parser.add_argument('--labels', type=str, required=False, default=None,
                               help='Id of the container')
    submit_parser.set_defaults(func=submit_job)
    jobs_parser = subparsers.add_parser('jobs', help="Get Jobs")
    jobs_parser.set_defaults(func=get_jobs)
    get_log_parser = subparsers.add_parser('get-logs', help="Get Logs")
    get_log_parser = subparsers.add_parser('get-logs', help="Cancel Job")

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)

    main()