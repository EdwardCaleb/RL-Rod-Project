import time
from optitrack_natnet_main.NatNetClient import NatNetClient
from optitrack_natnet_main.util import quaternion_to_euler


class OptiTrackClient:
    def __init__(self, client_address, server_address):
        self.positions = {}
        self.rotations = {}
        self.quaternions = {}

        self.client = NatNetClient()
        self.client.set_client_address(client_address)
        self.client.set_server_address(server_address)
        self.client.set_use_multicast(True)

        # Conectar callback
        self.client.rigid_body_listener = self._rigid_body_callback

    def _rigid_body_callback(self, rigid_id, position, rotation_quaternion):
        self.positions[rigid_id] = position
        self.quaternions[rigid_id] = rotation_quaternion

        # Euler (opcional)
        rotx, roty, rotz = quaternion_to_euler(rotation_quaternion)
        self.rotations[rigid_id] = (rotx, roty, rotz)

    def start(self):
        return self.client.run()

    def get_position(self, rigid_id):
        return self.positions.get(rigid_id, None)

    def get_rotation(self, rigid_id):
        return self.rotations.get(rigid_id, None)

    def get_quaternion(self, rigid_id):
        return self.quaternions.get(rigid_id, None)

    def get_pose(self, rigid_id):
        pos = self.get_position(rigid_id)
        rot = self.get_rotation(rigid_id)
        return pos, rot
    
    def get_full_pose(self, rigid_id):
        pos = self.positions.get(rigid_id, None)
        quat = self.quaternions.get(rigid_id, None)
        return pos, quat