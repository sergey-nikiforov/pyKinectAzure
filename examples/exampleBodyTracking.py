import sys
import cv2
import json
import numpy as np

sys.path.insert(1, "../")
import pykinect_azure as pykinect

serial_num_to_camera_name = {
    "000840215012": "a1",
    "000872415012": "a2",
    "000844515012": "a3",
    "000166110212": "b1",
    "000543214112": "b2",
    "000132112912": "b3",
    "000750215012": "c1",
    "000705215012": "c2",
    "000693315012": "c3",
}

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
    # print(device_config)

    for device_idx in range(9):
        # Start device
        device = pykinect.start_device(config=device_config, device_index=device_idx)
        camera_name = serial_num_to_camera_name.get(device.get_serialnum(), "x")

        # Start body tracker
        bodyTracker = pykinect.start_body_tracker()

        for i in range(20):
            print(f"device: {device_idx} camera_name: {camera_name} frame_num: {i}")

            # Get capture
            capture = device.update()

            # Get body tracker frame
            body_frame = bodyTracker.update()

            if body_frame.get_num_bodies():
                print(f"Detected {body_frame.get_num_bodies()} bodies")
                joints, confidence_levels = body_frame.get_3d_joints(0)
                d = {"joints_3d": joints, "confidence_levels": confidence_levels}
                json.dump(d, open(f"output/joints_{camera_name}.json", "w"))

                ret, depthmap = capture.get_depth_image()
                np.save(f"output/depthmap_{camera_name}.npy", depthmap)

                # Get the color depth image from the capture
                ret, depth_color_image = capture.get_colored_depth_image()

                # Get the colored body segmentation
                ret, body_image_color = body_frame.get_segmentation_image()

                if not ret:
                    continue

                # Combine both images
                combined_image = cv2.addWeighted(
                    depth_color_image, 0.6, body_image_color, 0.4, 0
                )

                # Draw the skeletons
                combined_image = body_frame.draw_bodies(combined_image)

                # Overlay body segmentation on depth image
                cv2.imwrite(f"output/bodysegm_{camera_name}.png", combined_image)
