import datetime

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from people_detection_ros2_msg.msg import People, BoundingBox
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from people_detection_ros2.people_detection_wrapper import PeopleDetectionWrapper


class PeopleDetectionNode(Node):
    people_detection_wrapper: PeopleDetectionWrapper

    def __init__(self):
        super().__init__('people_detection_node')

        # ros params
        is_debug_mode_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                       description='If true, run debug mode.')
        self.declare_parameter('is_debug_mode', False, is_debug_mode_descriptor)
        self.is_debug_mode: bool = self.get_parameter("is_debug_mode").get_parameter_value().bool_value
        trained_model_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                       description='The path of trained model.')
        self.declare_parameter('trained_model_path', '/model/path', trained_model_descriptor)
        trained_model_path: str = self.get_parameter("trained_model_path").get_parameter_value().string_value
        score_threshold_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                         description='Threshold of detection score.')
        self.declare_parameter('score_threshold', '/model/path', score_threshold_descriptor)
        score_threshold: float = self.get_parameter("score_threshold").get_parameter_value().double_value
        is_image_compressed_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                             description='Is input image compressed?')
        self.declare_parameter('is_image_compressed', False, is_image_compressed_descriptor)
        is_image_compressed: bool = self.get_parameter("is_image_compressed").get_parameter_value().bool_value
        image_node_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                    description='The node name of input image.')
        self.declare_parameter('image_node', '/image', image_node_descriptor)
        image_node: str = self.get_parameter("image_node").get_parameter_value().string_value

        self.people_detection_wrapper = PeopleDetectionWrapper(trained_model_path, score_threshold, self.is_debug_mode)
        self.bridge = CvBridge()

        # show info
        self.get_logger().info('IsDebugMode : ' + str(self.is_debug_mode))
        self.get_logger().info('TrainedModelPath : ' + trained_model_path)
        self.get_logger().info('ScoreThreshold : ' + str(score_threshold))
        self.get_logger().info('ImageNode : ' + image_node)
        self.get_logger().info('IsImageCompressed : ' + str(is_image_compressed))

        self._publisher = self.create_publisher(People, '/people_detection', 10)

        if is_image_compressed:
            self.subscription = self.create_subscription(CompressedImage, image_node,
                                                         self.get_img_compressed_callback, 10)
        else:
            self.subscription = self.create_subscription(Image, image_node,
                                                         self.get_img_callback, 10)

        # FPS計測
        self.frame_count = 0
        self.measurement_count = 10
        self.before_frame = 0
        self.fps = 0
        self.tm = cv2.TickMeter()
        self.tm.start()

    def publish_from_img(self, img: np.ndarray, timestamp: Time):
        self.frame_count += 1

        masked_img, boxes = self.people_detection_wrapper.detect(img)

        people = People()
        people.mask = self.bridge.cv2_to_compressed_imgmsg(masked_img, 'png')

        for box in boxes:
            bounding_box = BoundingBox()
            bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height = list(map(int, box))
            people.person_bouding_box_list.append(bounding_box)

        people.header.stamp = timestamp
        self._publisher.publish(people)

        if self.is_debug_mode:
            result_img = np.zeros(img.shape[:2]).astype(np.uint8)

            if self.frame_count % self.measurement_count == 0:
                self.tm.stop()
                self.fps = (self.frame_count - self.before_frame) / self.tm.getTimeSec()
                self.before_frame = self.frame_count
                self.tm.reset()
                self.tm.start()

            debug_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_img, "frame = " + str(self.frame_count), (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
            cv2.putText(debug_img, 'FPS: {:.2f}'.format(self.fps),
                        (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

            combine_img = np.zeros_like(img)
            for i in range(img.shape[2]):
                combine_img[:, :, i] = np.where(masked_img > 0, img[:, :, i], 0)
            for box in boxes:
                x, y, width, height = box
                cv2.rectangle(combine_img, (x, y), (x + width, y + height), (0, 0, 255), thickness=1)

            cv2.imshow('Result', cv2.hconcat([img, debug_img, combine_img]))
            cv2.waitKey(1)

            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

    def get_img_callback(self, image_raw: Image) -> None:
        try:
            image: np.ndarray = self.bridge.imgmsg_to_cv2(image_raw)
            self.publish_from_img(image, image_raw.header.stamp)
        except Exception as err:
            self.get_logger().error(err)

    def get_img_compressed_callback(self, image_raw: CompressedImage) -> None:
        try:
            image: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(image_raw)
            self.publish_from_img(image, image_raw.header.stamp)

        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    people_detection_node = PeopleDetectionNode()

    try:
        rclpy.spin(people_detection_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        print()
        people_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
