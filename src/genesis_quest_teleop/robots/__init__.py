from .franka import FrankaAdapter
from .openarm import OpenArmAdapter


def create_robot_adapter(config):
    robot_type = config["robot"]["type"]
    if robot_type == "franka":
        return FrankaAdapter(config)
    if robot_type == "openarm":
        return OpenArmAdapter(config)
    raise ValueError(f"unsupported robot.type: {robot_type}")
