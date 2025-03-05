#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <tf2_eigen/tf2_eigen.hpp>


class LidarPointCloudNode : public rclcpp::Node {
public:
    LidarPointCloudNode()
        : Node("lidar_pointcloud_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        // Subscribing to the LaserScan topic
        laser_scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&LidarPointCloudNode::laserScanCallback, this, std::placeholders::_1));
        
        // Publisher for PointCloud2
        point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cloud", 10);

        RCLCPP_INFO(this->get_logger(), "Lidar PointCloud Node Initialized");
    }


private:
    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        try {
            // Get transform from lidar frame to a fixed frame (e.g., "base_link")
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf_buffer_.lookupTransform("laser", msg->header.frame_id, msg->header.stamp);

            // Convert LaserScan to PointCloud2
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (size_t i = 0; i < msg->ranges.size(); ++i) {
                if (std::isfinite(msg->ranges[i])) {
                    float angle = msg->angle_min + i * msg->angle_increment;
                    pcl::PointXYZ point;
                    point.x = msg->ranges[i] * cos(angle);
                    point.y = msg->ranges[i] * sin(angle);
                    point.z = 0.0;  // 2D lidar
                    cloud->points.push_back(point);
                }
            }

            // Transform point cloud to base frame
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform_stamped.transform);
            Eigen::Matrix4f transform_matrix = eigen_transform.matrix().cast<float>();
            pcl::transformPointCloud(*cloud, *transformed_cloud, transform_matrix);

            // Convert to ROS2 PointCloud2 message
            sensor_msgs::msg::PointCloud2 output;
            pcl::toROSMsg(*transformed_cloud, output);
            output.header.frame_id = "laser";
            output.header.stamp = msg->header.stamp;

            // Publish PointCloud2
            point_cloud_pub_->publish(output);
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Transform failure: %s", ex.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarPointCloudNode>());
    rclcpp::shutdown();
    return 0;
}
