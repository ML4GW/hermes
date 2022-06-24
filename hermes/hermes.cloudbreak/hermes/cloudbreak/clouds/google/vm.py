import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import yaml
from google.cloud import compute_v1 as compute
from google.oauth2.service_account import Credentials as GoogleCredentials
from requests import HTTPError

from hermes.cloudbreak.clouds.base.resource import ResourceMeta
from hermes.cloudbreak.clouds.base.vm import VMInstance, VMManager
from hermes.cloudbreak.clouds.google.utils import Credentials, make_credentials
from hermes.cloudbreak.utils import snakeify


class GoogleVMClient:
    def __init__(
        self,
        credentials: Credentials = None,
    ):
        if credentials is not None:
            if isinstance(credentials, str):
                credentials = make_credentials(credentials)
            self._client = compute.InstancesClient.from_service_account_json(
                credentials
            )
        else:
            self._client = compute.InstancesClient()

    @property
    def project(self):
        return self._client._transport._credentials._project_id

    @property
    def client(self):
        return self

    def make_request(self, request, **kwargs):
        request_type = re.sub(
            "Instances?Request", "", request.__class__.__name__
        )

        request_fn_name = snakeify(request_type)
        request_fn = getattr(self._client, request_fn_name)

        # self.throttle()
        return request_fn(request=request, **kwargs)


class VMNotFound(Exception):
    pass


class VMBadRequest(Exception):
    """Dummy exception for catching bad requests to the compute service"""

    pass


class GoogleVMManager(VMManager):
    def __init__(
        self,
        description: compute.Instance,
        credentials: Credentials = None,
    ):
        self.description = description
        super().__init__(description.name, GoogleVMClient(credentials))

    @property
    def project(self):
        return self.parent.project

    def create_one_vm(self, name):
        description = deepcopy(self.description)
        description.name = name
        return GoogleVMInstance.create(self, description)


class GoogleVMInstanceMeta(ResourceMeta):
    @property
    def resource_type(self):
        return compute.Instance


@dataclass
class GoogleVMInstance(VMInstance, metaclass=GoogleVMInstanceMeta):
    zone: str

    def __post_init__(self):
        self._has_startup_script = False
        super().__post_init__()

    @property
    def project(self):
        return self.parent.project

    @classmethod
    def create(
        cls,
        parent: Union[GoogleVMClient, GoogleVMManager],
        description: compute.Instance,
    ):
        # we can get some of the info we need implicitly
        # from the objects used to instantiate it:
        # the zone has to be specified in the machine type
        # in the vm description, and the project is set
        # by the credentials used to instantiate the client
        # used to create the instance
        zone = re.search(
            "(?<=zones/).+?(?=/)", description.machine_type
        ).group(0)

        # instatiate the object before we create it
        obj = super().create(cls, description, parent, zone=zone)

        # if our vm has a startup script, make a note of it
        # so we can make sure it completes before we call
        # the VM "ready"
        for item in description.metadata.items:
            if item.key == "startup-script":
                obj._has_startup_script = True

        request = compute.InsertInstanceRequest(
            instance_resource=description, project=obj.project, zone=obj.zone
        )

        # try to make the instance. If it already exists,
        # ignore it and move on TODO: is this the behavior we want?
        try:
            obj.client.make_request(request)
        except HTTPError as e:
            content = yaml.safe_load(e.response.content.decode("utf-8"))
            message = content["error"]["message"]
            if message != (
                "The resource 'projects/{}/zones/{}/instances/{}' "
                "already exists".format(obj.project, obj.zone, obj.name)
            ):
                raise RuntimeError(message) from e

        return obj

    @property
    def not_found(self):
        return VMNotFound

    @property
    def bad_request(self):
        return VMBadRequest

    def get_get_request(self):
        return compute.ListInstancesRequest(
            project=self.project, zone=self.zone, filter=f"name = {self.name}"
        )

    def get(self):
        get_request = self.get_get_request()
        response = self.client.make_request(get_request)

        try:
            return response.items[-1]
        except IndexError:
            raise ValueError(f"Couldn't get resource {self}")

    def get_delete_request(self):
        return compute.DeleteInstanceRequest(
            project=self.project, zone=self.zone, instance=self.name
        )

    def delete(self):
        delete_request = self.get_delete_request()

        # TODO: how to make Resource.not_found
        # compatible with the case where the
        # not found is not the error itself but
        # a specific way the exception is instantiated?
        try:
            self.client.make_request(delete_request)
        except HTTPError as e:
            if e.response.status_code == 404:
                raise self.not_found
            raise

    def is_deleted(self):
        try:
            self.get()
            return False
        except ValueError:
            return True

    @property
    def ip(self):
        if self._ip is not None:
            return self._ip

        config = self.get()
        if config.status == compute.Instance.Status.PROVISIONING:
            return None

        try:
            self._ip = config.network_interfaces[0].access_configs[0].nat_i_p
            return self._ip
        except (AttributeError, IndexError):
            # TODO: is this catch necessary given the
            # provisioning catch above?
            return None

    def is_ready(self):
        if self._is_ready:
            return True
        else:
            # first check to see if we even have an IP address yet
            # leave the problem of catching taking-too-long issues
            # to the timeout passed to self.wait_for
            # TODO: do we want to add a special check for this
            # taking too long, vs. the startup script taking too long?
            try:
                if self.ip is None:
                    # if not, we're not ready
                    return False
            except ValueError:
                # if we can't even be listed yet, we're definitely not ready
                return False

            # if we didn't run a startup script, then
            # we're good to go at this point
            if not self._has_startup_script:
                self._is_ready = True
                return True

            # otherwise, read the startup log to see if
            # there was a message indicating that setup
            # is completed
            try:
                output, _ = self.run("cat /var/log/daemon.log")
            except EOFError:
                # we tried to read while the file was being
                # written to, so ignore this and try again
                return False
            self._is_ready = "startup-script exit status 0" in output
            return self._is_ready


def make_simple_debian_instance_description(
    name: str,
    zone: str,
    vcpus: int,
    service_account: Union[compute.ServiceAccount, Credentials] = None,
    startup_script: Optional[str] = None,
):
    if service_account is not None and not isinstance(
        service_account, compute.ServiceAccount
    ):
        # we provided a service account argument that
        # wasn't an already instantiated ServiceAccount
        # object, so use the passed argument to infer
        # the service account email and use a standard scope
        if isinstance(service_account, str):
            if "@" in service_account:
                # we specified a service account email explicitly
                service_account_email = service_account
            else:
                # we passed the path to a service account json
                # key file, so load the file and read the
                # service account email from it
                try:
                    with open(service_account, "r") as f:
                        service_account_email = json.load(f)["client_email"]
                except FileNotFoundError:
                    # the file didn't exist, so we have no idea
                    # what this is supposed to be
                    raise ValueError(
                        "Unable to recognize service account string {}".format(
                            service_account
                        )
                    )
        elif isinstance(service_account, GoogleCredentials):
            # if we provided service account credentials,
            # we can read the email from them
            service_account_email = service_account._service_account_email
        else:
            # otherwise we have no way to infer a service
            # account email from whatever this is
            raise TypeError(
                "Unrecognized service account type {}".format(
                    type(service_account)
                )
            )

        # instatiate a service account with the inferred email
        # and a standard scope. If you want other scopes,
        # instatiate this somewhere else and specify those
        # scopes explicitly
        service_account = compute.ServiceAccount(
            email=service_account_email,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    if service_account is not None:
        # service_accounts arg expects a list
        service_account = [service_account]

    # add startup script as metadata if we specified one
    if startup_script is not None:
        metadata = compute.Metadata(
            items=[compute.Items(key="startup-script", value=startup_script)]
        )
    else:
        metadata = None

    # TODO: allow specification of IP address
    return compute.Instance(
        name=name,
        service_accounts=service_account,
        machine_type=f"zones/{zone}/machineTypes/n1-standard-{vcpus}",
        network_interfaces=[
            compute.NetworkInterface(
                access_configs=[compute.AccessConfig()],
            )
        ],
        disks=[
            compute.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute.AttachedDiskInitializeParams(
                    source_image=(
                        "projects/debian-cloud/global/images/family/debian-10"
                    )
                ),
            )
        ],
        metadata=metadata,
    )
