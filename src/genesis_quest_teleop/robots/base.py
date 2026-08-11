from abc import ABC, abstractmethod


class RobotAdapter(ABC):
    @property
    @abstractmethod
    def entity(self): ...
    @property
    @abstractmethod
    def ee_link(self): ...
    @property
    @abstractmethod
    def arm_dofs_idx(self): ...
    @property
    @abstractmethod
    def finger_dofs_idx(self): ...
    @abstractmethod
    def build(self, scene): ...
    @abstractmethod
    def initialize_after_scene_build(self): ...
