from typing import Optional

import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import PhysxSchema

class Wheellegged(Robot):
    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "Wheellegged",
            usd_path: Optional[str] = None,
            translation: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None, 
    ) -> None:
        """[summary]"""
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            #usd path
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path = prim_path,
            name = name,
            translation = translation,
            orientation = orientation,
            articulation contorller = None,
        )

        self._dof_names = [

        ]
    
    @property
    def dof_names(self):
        return self._dof_names
    
    def set_wheellegged_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64 / np.pi * 180)

    def prepare_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                if "_HIP" not in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)
