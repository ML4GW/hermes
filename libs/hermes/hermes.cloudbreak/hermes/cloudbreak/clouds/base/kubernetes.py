import abc
import time
from dataclasses import dataclass
from typing import Dict, Optional

from hermes.cloudbreak.clouds.base.resource import ManagerResource, Resource
from hermes.cloudbreak.kubernetes import K8sApiClient
from hermes.cloudbreak.kubernetes.base import DaemonSet, Deployment, Service


class ClusterManager(ManagerResource):
    @property
    def managed_resource_type(self):
        return Cluster

    @classmethod
    def create(cls, *args, **kwargs):
        raise TypeError(
            "Cannot call `create` factory method for ClusterManager"
        )


@dataclass
class Cluster(ManagerResource):
    def __post_init__(self):
        self._k8s_client = None
        super().__post_init__()

    @property
    def managed_resource_type(self):
        return NodePool

    @property
    def k8s_client(self):
        # try to create the client this way because otherwise we
        # would need to wait until the cluster is ready at
        # initialization time in order to get the endpoint. If you're
        # not going to call `wait_for(cluster.is_ready)`, make sure to
        # wrap this in a catch for a RuntimeError
        # TODO: is it worth starting to introduce custom errors here
        # to make catching more intelligible?
        if self._k8s_client is None:
            self._k8s_client = K8sApiClient(self)
        return self._k8s_client

    def deploy(
        self,
        file: str,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        exists_ok: bool = True,
        **kwargs,
    ):
        return self.k8s_client.create_from_yaml(
            file, repo, branch, exists_ok, **kwargs
        )

    def deployments(self) -> Dict[str, Dict[str, Deployment]]:
        return dict(self.k8s_client._deployments)

    def services(self) -> Dict[str, Dict[str, Service]]:
        return dict(self.k8s_client._services)

    def daemon_sets(self) -> Dict[str, Dict[str, DaemonSet]]:
        return dict(self.k8s_client._daemon_sets)

    @abc.abstractproperty
    def token(self) -> bytes:
        pass

    @abc.abstractmethod
    def refresh_credentials(self) -> None:
        pass

    @abc.abstractmethod
    def deploy_gpu_drivers(self) -> None:
        pass


@dataclass
class NodePool(Resource):
    wait: Optional[float] = None

    def __post_init__(self):
        self._init_time = time.time()

    @abc.abstractmethod
    def check_stockout(self, status, reason):
        pass

    def raise_bad_status(self, status, reason):
        if not self.check_stockout(status, reason):
            super().raise_bad_status(status, reason)
        elif self.wait is None or (time.time() > self._init_time) < self.wait:
            # if we indicated not to wait or have exhausted the
            # amount of time we were willing to wait, raise an error
            raise RuntimeError(
                f"Resource {self} encountered stockout "
                "on creation and timed out"
            )
