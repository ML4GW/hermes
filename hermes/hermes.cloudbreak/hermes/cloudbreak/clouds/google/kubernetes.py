import math
import os
from typing import Optional, Union

import google
from google.cloud import container_v1 as container

from hermes.cloudbreak.clouds.base.kubernetes import (
    Cluster,
    ClusterManager,
    NodePool,
)
from hermes.cloudbreak.clouds.base.resource import (
    Client,
    ManagerResource,
    Resource,
    ResourceMeta,
)
from hermes.cloudbreak.clouds.google.utils import (
    Credentials,
    make_credentials,
    refresh,
)
from hermes.cloudbreak.utils import snakeify


class GoogleClient(Client):
    def __init__(
        self,
        credentials: Credentials = None,
        throttle_secs: float = 1.0,
    ):
        if isinstance(credentials, str):
            credentials = make_credentials(credentials)
        self._client = container.ClusterManagerClient(credentials=credentials)

        super().__init__(throttle_secs)

    def make_request(self, request, **kwargs):
        request_type = request.__class__.__name__.replace("Request", "")
        request_fn_name = snakeify(request_type)
        request_fn = getattr(self._client, request_fn_name)

        self.throttle()
        return request_fn(request=request, **kwargs)


class GoogleClusterMeta(ResourceMeta):
    @property
    def resource_type(self):
        return container.Cluster


class GoogleNodePoolMeta(ResourceMeta):
    @property
    def resource_type(self):
        return container.NodePool


class GoogleResource(Resource):
    @property
    def not_found(self):
        return google.api_core.exceptions.NotFound

    @property
    def bad_request(self):
        return google.api_core.exceptions.BadRequest

    def raise_bad_status(self, response):
        super().raise_bad_status(response.status, response.conditions)

    @classmethod
    def create(cls, resource, parent, **kwargs):
        obj = super().create(cls, resource, parent, **kwargs)

        create_request_cls = getattr(
            container, f"Create{resource.__class__.__name__}Request"
        )
        resource_type = snakeify(resource.__class__.__name__)

        kwargs = {resource_type: resource, "parent": str(parent)}
        create_request = create_request_cls(**kwargs)
        try:
            obj.client.make_request(create_request)
        except google.api_core.exceptions.AlreadyExists:
            pass

        return obj

    def get_delete_request(self):
        resource_type = self.__class__.__name__.replace("Google", "")
        delete_request_cls = getattr(
            container, f"Delete{resource_type}Request"
        )
        return delete_request_cls(name=str(self))

    def get_get_request(self):
        resource_type = self.__class__.__name__.replace("Google", "")
        get_request_cls = getattr(container, f"Get{resource_type}Request")
        return get_request_cls(name=str(self))

    def is_ready(self) -> bool:
        # TODO: should we return False instead of raising
        # an error for statuses greater than 3?
        response = self.get(timeout=5)
        if response.status == 2:
            return True
        elif response.status == 3:
            return False
        elif response.status > 3:
            self.raise_bad_status(response)
        return False

    def is_deleted(self) -> bool:
        """
        check if a submitted delete request has completed
        """
        try:
            response = self.get(timeout=5)
        except ValueError as e:
            if str(e) != f"Couldn't get resource {self}":
                raise
            # couldn't find the resource, so assume
            # the deletion went off swimmingly
            return True
        if response.status > 5:
            self.raise_bad_status(response)
        return False

    def __str__(self):
        resource_type = self.__class__.__name__.replace("Google", "")
        camel = resource_type[0].lower() + resource_type[1:]
        return str(self.parent) + "/{}/{}".format(camel, self.name)


class GoogleNodePool(GoogleResource, NodePool, metaclass=GoogleNodePoolMeta):
    def check_stockout(self, status, reason):
        return reason[0].code == container.StatusCondition.Code.GCE_STOCKOUT


class GoogleManagerResource(ManagerResource, GoogleResource):
    @property
    def _mrt(self):
        return self.managed_resource_type.__name__.replace("Google", "") + "s"

    def get_list_request(self):
        list_request_cls = getattr(container, f"List{self._mrt}Request")
        return list_request_cls(parent=str(self))

    def parse_list_response(self, response):
        resources = getattr(response, snakeify(self._mrt))
        return [r.name for r in resources]


class GoogleCluster(
    GoogleManagerResource, Cluster, metaclass=GoogleClusterMeta
):
    @property
    def managed_resource_type(self):
        return GoogleNodePool

    @property
    def token(self):
        return self.client._client._transport._credentials.token

    def refresh_credentials(self) -> str:
        refresh(self.client._client._transport._credentials)

    def deploy_gpu_drivers(self) -> None:
        drivers_daemon_set = self.deploy(
            "nvidia-driver-installer/cos/daemonset-preloaded.yaml",
            repo="GoogleCloudPlatform/container-engine-accelerators",
            branch="master",
            exists_ok=True,
        )[0]
        drivers_daemon_set.wait_for_ready()


class GoogleClusterManager(GoogleManagerResource, ClusterManager):
    def __init__(
        self,
        zone: str,
        project: Optional[str] = None,
        credentials: Union[str, Credentials, None] = None,
        throttle_secs: float = 1.0,
    ) -> None:
        if project is None and credentials is None:
            # try to infer the credentials from the environment
            try:
                credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                if not os.path.exists(credentials):
                    raise KeyError
            except KeyError:
                raise ValueError(
                    "Must specify a project to associated with the cluster "
                    "or a set of credentials from which to infer it"
                )

        # if we specified string credentials, or inferred
        # them in the last clause, instantiate a credentials
        # object for consistency
        if credentials is not None and isinstance(credentials, str):
            # instantiate a credentials object for consistency
            credentials = make_credentials(credentials)

        # if we didn't specify a project but have been able
        # to infer credentials at this point, grab the project
        # id from those credentials
        if project is None:
            project = credentials._project_id

        parent = GoogleClient(credentials, throttle_secs)
        name = f"projects/{project}/locations/{zone}"
        super().__init__(name, parent)

    @property
    def managed_resource_type(self):
        return GoogleCluster

    def __str__(self):
        return self.name


def create_gpu_node_pool_config(
    vcpus: int, gpus: int, gpu_type: str, **kwargs
) -> container.NodeConfig:
    if (math.log2(vcpus) % 1 != 0 and vcpus != 96) or vcpus > 96:
        raise ValueError(f"Can't configure node pool with {vcpus} vcpus")

    if gpus < 1 or gpus > 8:
        raise ValueError(f"Can't configure node pool with {gpus} gpus")

    if gpu_type not in ["t4", "v100", "p100", "p4", "k80"]:
        raise ValueError(
            "Can't configure n1 standard node pool "
            f"with unknown gpu type {gpu_type}"
        )

    return container.NodeConfig(
        machine_type=f"n1-standard-{vcpus}",
        oauth_scopes=[
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring",
            "https://www.googleapis.com/auth/service.management.readonly",
            "https://www.googleapis.com/auth/servicecontrol",
            "https://www.googleapis.com/auth/trace.append",
        ],
        accelerators=[
            container.AcceleratorConfig(
                accelerator_count=gpus,
                accelerator_type=f"nvidia-tesla-{gpu_type}",
            )
        ],
        **kwargs,
    )
