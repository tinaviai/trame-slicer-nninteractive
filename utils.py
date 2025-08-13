# -*- coding:utf-8 -*-
import torch
import os
import numpy as np

from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = ".nninteractive_weights"  # Specify the download directory


class PromptManager:
    """
    Manages the image, target tensor, and runs inference sessions for point, bbox,
    lasso, and scribble interactions.
    """

    def __init__(self):
        self.img = None
        self.target_tensor = None

        self.download_weights()
        self.session = self.make_session()

    def download_weights(self):
        """
        Downloads only the files matching 'MODEL_NAME/*' into DOWNLOAD_DIR.
        """
        snapshot_download(
            repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
        )

    def make_session(self):
        """
        Creates an nnInteractiveInferenceSession, points it at the downloaded model.
        """
        session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=True,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session

    def set_image(self, input_image):
        """
        Loads the user-provided 3D image into the session, resets interactions.
        """
        self.session.reset_interactions()

        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)

        print("self.img.shape:", self.img.shape)

        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(
            self.img.shape[1:], dtype=torch.uint8
        )  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)

    def set_segment(self, mask):
        """
        Sets or resets a segmentation (mask) on the server side.
        If mask is empty, resets the session's interactions.
        """
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(
                self.img.shape[1:], dtype=torch.uint8
            )  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask)

    def add_point_interaction(self, point_coordinates, include_interaction):
        """
        Process a point-based interaction (positive or negative).
        """
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction
        )

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_bbox_interaction(
            self, outer_point_one, outer_point_two, include_interaction
    ):
        """
        Process bounding box-based interaction.
        """
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)

        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)

        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]

        # Call the session's bounding box interaction function.
        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_lasso_interaction(self, mask, include_interaction):
        """
        Process lasso-based interaction using a 3D mask.
        """
        print("Lasso mask received with shape:", mask.shape)
        self.session.add_lasso_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_scribble_interaction(self, mask, include_interaction):
        """
        Process scribble-based interaction using a 3D mask.
        """
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()
