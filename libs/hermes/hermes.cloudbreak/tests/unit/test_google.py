import os

import pytest

from hermes.cloudbreak.clouds import google as cb


def test_cluster_manager(zone="us-central1-f"):
    manager = cb.ClusterManager(zone=zone)

    creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
    with pytest.raises(ValueError):
        manager = cb.ClusterManager(zone=zone)  # noqa

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
