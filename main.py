import cv2 as cv2
import pyrealsense2 as rs
import numpy as np


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    hole_filter = rs.hole_filling_filter(2)

    w = 848
    h = 480
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)

    # Note in the example code, cfg is misleadingly called "profile" but cfg is a better name
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    h_min = 113
    h_max = 136
    v_min = 16
    v_max = 142
    s_min = 86
    s_max = 227

    def set_h_min(val):
        nonlocal h_min
        h_min = val

    def set_h_max(val):
        nonlocal h_max
        h_max = val

    def set_s_min(val):
        nonlocal s_min
        s_min = val

    def set_s_max(val):
        nonlocal s_max
        s_max = val

    def set_v_min(val):
        nonlocal v_min
        v_min = val

    def set_v_max(val):
        nonlocal v_max
        v_max = val

    cv2.namedWindow("thresh")
    cv2.createTrackbar("HMIN", "thresh", h_min, 180, set_h_min)
    cv2.createTrackbar("HMAX", "thresh", h_max, 180, set_h_max)
    cv2.createTrackbar("SMIN", "thresh", s_min, 255, set_s_min)
    cv2.createTrackbar("SMAX", "thresh", s_max, 255, set_s_max)
    cv2.createTrackbar("VMIN", "thresh", v_min, 255, set_v_min)
    cv2.createTrackbar("VMAX", "thresh", v_max, 255, set_v_max)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frame = align.process(frames)
            depth_frame = aligned_frame.get_depth_frame()
            depth_frame = hole_filter.process(depth_frame)
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # convert to hsv
            hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # blur
            blur = cv2.medianBlur(hsv_img, 5)

            # binary filter
            thresh_img = cv2.inRange(
                blur, (h_min, s_min, v_min), (h_max, s_max, v_max))

            # erosion/dilation for cleanup
            eroded_img = cv2.erode(thresh_img, np.ones((5, 5)))
            dilated_img = cv2.dilate(eroded_img, np.ones((10, 10)))
            closed_img = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_RECT, (3, 3)), iterations=10)

            # contours
            contours, hierarchy = cv2.findContours(
                closed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours = list(contours)
            contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
            if len(contours) > 0:
                pen_contour = contours[0]

                # get center of mass
                M = cv2.moments(pen_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.drawContours(
                    color_image, [pen_contour], -1, (0, 255, 0), 3)
                cv2.circle(color_image, (cX, cY), 3, (0, 0, 255), 3)

                total = 0
                depth = depth_image[cY][cX]
                # for r in range(max(cY - 3, 0), min(cY+3, h)):
                #     for c in range(max(cX - 3, 0), min(cX + 3, w)):
                #         if depth_image[r][c]:
                #             depth += depth_image[r][c]
                #             total += 1

                # if total == 0:
                #     depth = 0
                # else:
                #     depth = depth / total
                point = rs.rs2_deproject_pixel_to_point(
                    intr, [cX, cY], depth)

                text = f"({round(point[0])}, {round(point[1])}, {round(point[2])})"
                color_image = cv2.putText(
                    color_image,
                    text,
                    (cX+3, cY+3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    3, cv2.LINE_AA
                )

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('RealSense', color_image)
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('HSV', hsv_img)
            cv2.imshow('thresh', thresh_img)
            cv2.imshow("closed", closed_img)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
