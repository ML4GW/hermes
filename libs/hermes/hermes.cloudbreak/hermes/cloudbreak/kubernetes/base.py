import abc
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import kubernetes
from kubernetes.utils.create_from_yaml import FailToCreateError
from urllib3.exceptions import MaxRetryError

from hermes.cloudbreak.utils import snakeify, wait_for

if TYPE_CHECKING:
    from hermes.cloudbreak.kubernetes import K8sApiClient


@dataclass
class Resource(abc.ABC):
    _client: "K8sApiClient"
    name: str
    namespace: str = "default"

    MAX_RETRY_GRACE_SECONDS = 300
    STATUS_AVAILABLE_GRACE_SECONDS = 10

    @classmethod
    def create(cls, client, config):
        if config["kind"] == "Deployment":
            cls = Deployment
        elif config["kind"] == "Service":
            cls = Service
        elif config["kind"] == "DaemonSet":
            cls = DaemonSet
        else:
            raise ValueError(
                "Resource kind {} not supported yet".format(config["kind"])
            )

        metadata = config["metadata"]
        obj = cls(client, metadata["name"], metadata["namespace"])

        create_fn = partial(
            kubernetes.utils.create_from_dict,
            k8s_client=client._client,
            data=config,
        )
        response = obj._make_a_request(create_fn)
        if response is None:
            raise MaxRetryError
        return obj

    def __post_init__(self):
        self._creation_time = time.time()
        self._unavailable = False
        self._unavailable_time = None

    @abc.abstractproperty
    def client(self):
        pass

    def _make_a_request(self, request_fn, do_raise=False):
        try:
            # try to make the request
            return request_fn()
        except (
            kubernetes.client.exceptions.ApiException,
            FailToCreateError,
        ) as e:
            try:
                # create from yaml wraps around API exceptions,
                # so grab the underlying exception here first
                status = e.api_exceptions[0].status
            except AttributeError:
                status = e.status

            if status != 401:
                raise

            if not do_raise:
                self._client.cluster.refresh_credentials()
                self._client._client.configuration.api_key[
                    "authorization"
                ] = self._client.cluster.token

                # try the request again with do_raise set to
                # true to indicate that these credentials just
                # don't have access to this cluster
                return self._make_a_request(request_fn, do_raise=True)
            else:
                # if do_raise is set, indicate that the request
                # is unauthorized
                raise RuntimeError("Unauthorized request to cluster")

        except MaxRetryError:
            # sometimes this error can get raised if the master nodes
            # of the cluster are busy doing something. Return None
            # to indicate this is happening but give things a few
            # minutes to get back to normal
            if not self._unavailable:
                self._unavailable = True
                self._unavailable_time = time.time()
            elif (
                time.time() - self._unavailable_time
            ) < self.MAX_RETRY_GRACE_SECONDS:
                raise RuntimeError(
                    "Deployment {} has been unavailable for {} seconds".format(
                        self.name, self.MAX_RETRY_GRACE_SECONDS
                    )
                )
            return None
        except Exception as e:
            print(type(e), e)
            raise

    def get(self):
        resource_type = snakeify(self.__class__.__name__)
        get_fn = partial(
            getattr(self.client, f"read_namespaced_{resource_type}_status"),
            name=self.name,
            namespace=self.namespace,
        )
        try:
            response = self._make_a_request(get_fn)
            self._unavailable = False
            return response
        except kubernetes.client.ApiException as e:
            if e.status == 404:
                raise RuntimeError(f"{self.message} no longer exists")
            raise

    def delete(self):
        resource_type = snakeify(self.__class__.__name__)
        delete_fn = partial(
            getattr(self.client, f"delete_namespaced_{resource_type}_status"),
            name=self.name,
            namespace=self.namespace,
        )
        return self._make_a_request(delete_fn)

    @abc.abstractmethod
    def is_ready(self):
        pass

    def wait_for_ready(self):
        wait_for(
            self.is_ready,
            f"Waiting for {self.message} to become ready",
        )

    def submit_delete(self):
        try:
            response = self.delete()
            return response is not None
        except kubernetes.client.ApiException as e:
            if e.status == 404:
                return True
            raise

    def is_deleted(self):
        try:
            self.get()
        except RuntimeError as e:
            if str(e).endswith("no longer exists"):
                return True
            raise
        else:
            return False

    def remove(self):
        if not self.submit_delete():
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

        # TODO: remove this from self._client resources?

    @property
    def message(self):
        resource_type = snakeify(self.__class__.__name__).replace("_", " ")
        return " ".join([resource_type, self.name])


class Deployment(Resource):
    @property
    def client(self):
        return kubernetes.client.AppsV1Api(self._client._client)

    # TODO: custom wait that clocks that the number of available instances
    def is_ready(self):
        response = self.get()
        if response is None:
            return False

        conditions = response.status.conditions
        if conditions is None:
            return False
        statuses = {i.type: eval(i.status) for i in conditions}

        if len(statuses) == 0 and (
            (time.time() - self._creation_time)
            > self.STATUS_AVAILABLE_GRACE_SECONDS
        ):
            raise RuntimeError(
                "Deployment {} has gone {} seconds with no "
                "available status information".format(
                    self.name, self.STATUS_AVAILABLE_GRACE_SECONDS
                )
            )

        try:
            if statuses["Available"]:
                return True
        except KeyError:
            try:
                if not statuses["Progressing"]:
                    raise RuntimeError(f"{self.message} stopped progressing")
            except KeyError:
                return False

    def scale(self, replicas: int):
        response = self.get()
        if response is None:
            return False

        response.spec.replicas = replicas
        scale_fn = partial(
            self.client.patch_namespaced_deployment_scale,
            name=self.name,
            namespace=self.namespace,
            body=response,
        )
        return self._make_a_request(scale_fn)


@dataclass
class Service(Resource):
    """Really represents specifically a LoadBalancer"""

    def __post_init__(self):
        self._ip = None

    @property
    def client(self):
        return kubernetes.client.CoreV1Api(self._client._client)

    @property
    def ip(self):
        if self._ip is None:
            response = self.get()
            if response is None:
                return None

            try:
                self._ip = response.status.load_balancer.ingress[0].ip
            except TypeError:
                return None
        return self._ip

    def is_ready(self):
        # server is considered ready once it has a public IP address
        return self.ip is not None


class DaemonSet(Resource):
    @property
    def client(self):
        return kubernetes.client.AppsV1Api(self._client._client)

    def is_ready(self):
        response = self.get()
        if response is None:
            return False

        status = response.status
        return status.desired_number_scheduled == status.number_ready
