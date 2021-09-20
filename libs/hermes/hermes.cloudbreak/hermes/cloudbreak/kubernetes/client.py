import re
from base64 import b64decode
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, List, Optional

import kubernetes
import requests
import yaml
from urllib3.exceptions import MaxRetryError

from hermes.cloudbreak.kubernetes.base import (
    DaemonSet,
    Deployment,
    Resource,
    Service,
)
from hermes.cloudbreak.utils import snakeify

if TYPE_CHECKING:
    from hermes.cloudbreak.clouds.base.kubernetes import Cluster


class K8sApiClient:
    def __init__(self, cluster: "Cluster"):
        # TODO: generalize the initialization putting methods
        # on the `cluster` object to return the appropriate
        # information
        try:
            response = cluster.get()
        except requests.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Cluster {cluster} not currently deployed")
            raise

        # create configuration using bare minimum info
        configuration = kubernetes.client.Configuration()
        configuration.host = f"https://{response.endpoint}"

        with NamedTemporaryFile(delete=False) as ca_cert:
            certificate = response.master_auth.cluster_ca_certificate
            ca_cert.write(b64decode(certificate))

        cluster.refresh_credentials()
        configuration.ssl_ca_cert = ca_cert.name
        configuration.api_key_prefix["authorization"] = "Bearer"
        configuration.api_key["authorization"] = cluster.token.decode()

        # return client instantiated with configuration
        self.cluster = cluster
        self._client = kubernetes.client.ApiClient(configuration)

        # look in some default locations for existing resources
        self._deployments = defaultdict(dict)
        self._services = defaultdict(dict)
        self._daemon_sets = defaultdict(dict)

        self.get_resources_for_namespace("default")
        self.get_resources_for_namespace("kube-system")

    def _add_resource(self, resource):
        resources = getattr(
            self, "_" + snakeify(resource.__class__.__name__) + "s"
        )
        resources[resource.namespace][resource.name] = resource

    def get_resources_for_namespace(self, namespace: str):
        apps_client = kubernetes.client.AppsV1Api(self._client)
        for deployment in apps_client.list_namespaced_deployment(
            namespace
        ).items:
            deployment = Deployment(self, deployment.metadata.name, namespace)
            self._add_resource(deployment)

        for daemon_set in apps_client.list_namespaced_daemon_set(
            namespace
        ).items:
            daemon_set = DaemonSet(self, daemon_set.metadata.name, namespace)
            self._add_resource(daemon_set)

        core_client = kubernetes.client.CoreV1Api(self._client)
        for service in core_client.list_namespaced_service(namespace).items:
            service = Service(self, service.metadata.name, namespace)
            self._add_resource(service)

    def create_from_yaml(
        self,
        file: str,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        exists_ok: bool = True,
        **kwargs,
    ) -> List[Resource]:
        # get deploy file content either from
        # local file or from github repo. Parse
        # yaml go template wildcards with kwargs
        content = get_content(file, repo, branch, **kwargs)

        failures = []
        k8s_objects = []

        # loop through content explicitly so that we
        # can preserver order for returning things
        for yml_document in content.split("---\n"):
            if not yml_document:
                continue

            # load the string to a dict
            yml_document = yaml.safe_load(yml_document)

            try:
                # first try to create the resource, assuming
                # it doesn't exist
                created = Resource.create(self, yml_document)

                # add the created resource to our internal
                # mapping of resources, and append it to return
                self._add_resource(created)
                k8s_objects.append(created)
            except kubernetes.utils.FailToCreateError as failure:
                # kubernetes exception in creation. Keep track
                # of all that get raised and raise them at the end
                for exc in failure.api_exceptions:
                    if exc.status == 409 and exists_ok:
                        # if the exception was that the object already
                        # exists in the namespace and we indicated that
                        # this was ok, find a workaround to return the
                        # appropriate object
                        metadata = yml_document["metadata"]
                        kind = yml_document["kind"]

                        try:
                            # try to get an object that's already been
                            # created if one exists
                            resources = getattr(
                                self, "_" + snakeify(kind) + "s"
                            )
                            created = resources[metadata["namespace"]][
                                metadata["name"]
                            ]
                        except KeyError:
                            # do a lazy creation of the object using
                            # the metadata
                            if kind == "Deployment":
                                cls = Deployment
                            elif kind == "Service":
                                cls = Service
                            elif kind == "DaemonSet":
                                cls = DaemonSet
                            else:
                                raise ValueError(
                                    "Resource kind {} not supported ".format(
                                        kind
                                    )
                                )

                            # create the object via regular initialization
                            created = cls(
                                self, metadata["name"], metadata["namespace"]
                            )

                            # add the object to internal resource mapping
                            self._add_resource(created)

                        # either way append the object to our returns
                        k8s_objects.append(created)
                    else:
                        # in any other case, keep track of the exception
                        # to raise later
                        failures.append(exc)
            except MaxRetryError:
                raise RuntimeError(
                    "Cluster occupied and couldn't connect to create"
                )

        if failures:
            # raise any and all kubernetes failures at once if we have any
            raise kubernetes.utils.FailToCreateError(failures)

        # otherwise return all the relevant objects
        # that were specified in the yaml
        return k8s_objects


def get_content(
    file: str,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    **kwargs,
) -> str:
    if repo is not None:
        if branch is None:
            # if we didn't specify a branch, default to
            # trying main first but try master next in
            # case the repo hasn't changed yet
            branches = ["main", "master"]
        else:
            # otherwise just use the specified branch
            branches = [branch]

        for branch in branches:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                content = response.content.decode()
                break
            except Exception:
                pass
        else:
            raise ValueError(
                "Couldn't find file {} at github repo {} "
                "in branches {}".format(file, repo, ", ".join(branches))
            )
    else:
        with open(file, "r") as f:
            content = f.read()

    return sub_values(content, **kwargs)


def sub_values(content: str, **kwargs) -> str:
    """Sub wildcard values into string

    Hacky replacement for helm-style variable values in
    K8s YAML descriptions. Wildcards are specified in
    `content` via {{ .Values.<variable name> }}, and will
    be replaced by their value in `**kwargs`. If a variable
    value is not given a kwarg, or if a kwarg is passed
    without a relevant wildcard, a ValueError is raised
    """

    match_re = re.compile("(?<={{ .Values.)[a-zA-Z0-9]+?(?= }})")
    found = set()

    def replace_fn(match):
        varname = match_re.search(match.group(0)).group(0)
        found.add(varname)
        try:
            return str(kwargs[varname])
        except KeyError:
            raise ValueError(f"No value provided for wildcard {varname}")

    content = re.sub("{{ .Values.[a-zA-Z0-9]+? }}", replace_fn, content)

    missing = set(kwargs) - found
    if missing:
        raise ValueError(
            "Provided unused wildcard values: {}".format(", ".join(missing))
        )
    return content
