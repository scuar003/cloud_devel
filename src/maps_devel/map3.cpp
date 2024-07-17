#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <map>

// Alias
using namespace std::placeholders;
using PointT = pcl::PointXYZRGB;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PCLcloud = pcl::PointCloud<PointT>;

class PCLConverter : public rclcpp::Node
{
public:
    PCLConverter() : Node("pcl_converter_node"), tf_buffer(this->get_clock()), tf_listener(tf_buffer)
    {
        sub_pointcloud = create_subscription<PointCloud2>("/camera/depth/color/points", 10, std::bind(&PCLConverter::cloudProcessing, this, _1));
        pub_process_cloud = create_publisher<PointCloud2>("processed_points", 10);
        accumulated_cloud.reset(new PCLcloud);
    }

private:
    void cloudProcessing(const PointCloud2::SharedPtr msg)
    {
        PCLcloud::Ptr current_cloud(new PCLcloud);
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *current_cloud);

        // Transform the point cloud to the base_link frame
        geometry_msgs::msg::TransformStamped transform_stamped;
        try {
            transform_stamped = tf_buffer.lookupTransform("base_link", msg->header.frame_id, tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform %s to base_link: %s", msg->header.frame_id.c_str(), ex.what());
            return;
        }

        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() << transform_stamped.transform.translation.x,
                                  transform_stamped.transform.translation.y,
                                  transform_stamped.transform.translation.z;
        Eigen::Quaterniond q(transform_stamped.transform.rotation.w,
                             transform_stamped.transform.rotation.x,
                             transform_stamped.transform.rotation.y,
                             transform_stamped.transform.rotation.z);
        transform.rotate(q);
        pcl::transformPointCloud(*current_cloud, *current_cloud, transform);

        // Downsample and accumulate the point cloud
        downSample(current_cloud, filtered_cloud);
        *accumulated_cloud += *filtered_cloud;

        // Publish the processed point cloud
        PointCloud2 ros2_cloud;
        toROS2(accumulated_cloud, ros2_cloud);
        pub_process_cloud->publish(ros2_cloud);
    }

    void toROS2(const PCLcloud::Ptr &pcl_cloud, PointCloud2 &ros2_cloud) {
        pcl::toROSMsg(*pcl_cloud, ros2_cloud);
        ros2_cloud.header.frame_id = "base_link";
        ros2_cloud.header.stamp = this->get_clock()->now();
    }

    void downSample(const PCLcloud::Ptr &current_cloud, PCLcloud::Ptr &filtered_cloud) {
        std::map<Eigen::Vector3i, PointT, std::function<bool(Eigen::Vector3i, Eigen::Vector3i)>> voxel_map([](const Eigen::Vector3i &a, const Eigen::Vector3i &b) {
            return a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])));
        });

        float resolution = 0.01f;
        for (const auto &point : *current_cloud) {
            Eigen::Vector3i idx(static_cast<int>(point.x / resolution), static_cast<int>(point.y / resolution), static_cast<int>(point.z / resolution));
            auto &p = voxel_map[idx];
            if (p.a == 0) {  // initialize the point attributes on first access
                p.x = 0;
                p.y = 0;
                p.z = 0;
                p.r = 0;
                p.g = 0;
                p.b = 0;
                p.a = 0;
            }
            p.x += point.x;
            p.y += point.y;
            p.z += point.z;
            p.r += point.r;
            p.g += point.g;
            p.b += point.b;
            p.a += 1;
        }

        for (auto &kv : voxel_map) {
            PointT &p = kv.second;
            int count = p.a;  // 'a' is used as count
            if (count > 0) {  // Check to avoid division by zero
                p.x /= count;
                p.y /= count;
                p.z /= count;
                p.r /= count;
                p.g /= count;
                p.b /= count;
                filtered_cloud->push_back(p);   
            }
        }
    }

    rclcpp::Subscription<PointCloud2>::SharedPtr sub_pointcloud;
    rclcpp::Publisher<PointCloud2>::SharedPtr pub_process_cloud;
    PCLcloud::Ptr accumulated_cloud;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
