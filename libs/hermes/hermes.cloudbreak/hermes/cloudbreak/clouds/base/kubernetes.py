import abc
import time
from dataclasses import dataclass
from typing import Dict, Optional

from hermes.cloudbreak.kubernetes import K8sApiClient
from hermes.cloudbreak.kubernetes.base import DaemonSet, Deployment, Service
from hermes.cloudbreak.utils import snakeify, wait_for


@dataclass
class Client:
    throttle_secs: float = 1.0

    def __post_init__(self):
        self._last_request_time = time.time()

    @property
    def client(self):
        return self

    @property
    def name(self):
        return ""

    def throttle(self):
        while (time.time() - self._last_request_time) < self.throttle_secs:
            time.sleep(0.01)
        self._last_request_time = time.time()

    @abc.abstractmethod
    def make_request(self, request, **kwargs):
        pass


class ResourceMeta(abc.ABCMeta):
    @property
    def resource_type(self):
        raise NotImplementedError


@dataclass
class Resource(metaclass=ResourceMeta):
    name: str
    parent: "Resource"

    @property
    def client(self):
        return self.parent.client

    def raise_bad_status(self, status, reason):
        # TODOL will this be general?
        raise RuntimeError(
            "Resource {} reached status {} with reason {}".format(
                self, status, reason
            )
        )

    @abc.abstractproperty
    def not_found(self):
        pass

    @abc.abstractproperty
    def bad_request(self):
        pass

    def create(cls, resource, parent, **kwargs):
        if not isinstance(resource, cls.resource_type):
            raise TypeError(f"{cls.__name__} can't manage resource {resource}")
        return cls(resource.name, parent, **kwargs)

    @abc.abstractmethod
    def get_delete_request(self):
        pass

    def delete(self):
        return self.client.make_request(self.get_delete_request())

    @abc.abstractmethod
    def get_get_request(self):
        pass

    def get(self, timeout=None):
        get_request = self.get_get_request()
        try:
            return self.client.make_request(get_request, timeout=timeout)
        except self.not_found:
            raise ValueError(f"Couldn't get resource {self}")

    @abc.abstractmethod
    def is_ready(self) -> bool:
        pass

    def wait_for_ready(self):
        wait_for(
            self.is_ready,
            f"Waiting for {self.message} to become ready",
        )

    def submit_delete(self) -> bool:
        """
        Attempt to submit a delete request for a resource.
        Returns `True` if the request is successfully
        submitted or if the resource can't be found,
        and `False` if the request can't be submitted
        """
        try:
            self.delete()
            return True
        except self.not_found:
            # resource is gone, so we're good
            return True
        except self.bad_request:
            # Resource is tied up, so indicate that
            # the user will need to try again later
            return False

    @abc.abstractmethod
    def is_deleted(self) -> bool:
        """
        check if a submitted delete request has completed
        """
        pass

    def remove(self):
        if not self.submit_delete():
            # TODO: do some check to see if it's unable to accept
            # delete requests because it's already deleting,
            # or if e.g. because it's spinning up. In the latter case:
            wait_for(
                self.submit_delete,
                f"Waiting for {self.message} to become available to delete",
            )

        if not self.is_deleted():
            # give us a chance to not have to display the progress bar
            wait_for(self.is_deleted, f"Waiting for {self.message} to delete")
        else:
            # TODO: logging?
            print(f"Deleted {self.message}")

        # if this resource belongs to a manager,
        # remove from its collection of resources
        try:
            self.parent._resources.pop(self.name)
        except AttributeError:
            pass

    @property
    def message(self):
        resource_type = snakeify(self.__class__.__name__).replace("_", " ")
        return " ".join([resource_type, self.name])

    def __enter__(self):
        if not self.is_ready():
            self.wait_for_ready()
        return self

    def __exit__(self, *exc_args):
        self.remove()


@dataclass
class ManagerResource(Resource):
    def __post_init__(self):
        self._resources = {}

        list_request = self.get_list_request()
        try:
            response = self.client.make_request(list_request)
        except self.not_found:
            return

        for name in self.parse_list_response(response):
            self._resources[name] = self.managed_resource_type(name, self)

    @abc.abstractproperty
    def managed_resource_type(self):
        pass

    @abc.abstractmethod
    def get_list_request(self):
        pass

    @abc.abstractmethod
    def parse_list_response(self, response):
        pass

    @property
    def resources(self):
        return self._resources.copy()

    def add(self, resource):
        try:
            resource = self.managed_resource_type.create(resource, self)
        except TypeError:
            raise TypeError(f"{self} cannot manage resource {resource}")
        self._resources[resource.name] = resource
        return resource


class ClusterManager(ManagerResource):
    @property
    def managed_resource_type(self):
        return Cluster

    @classmethod
    def create(cls, *args, **kwargs):
        raise TypeError(
            "Cannot call `create` factor method for ClusterManager"
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

    def daemon_sets(self) -> Dict[str, dict[str, DaemonSet]]:
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
