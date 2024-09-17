#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using PointT = pcl::PointXYZRGB;
using PCLcloud = pcl::PointCloud<PointT>;

class PointCloudSaver : public rclcpp::Node {
    public:
        PointCloudSaver() : Node("pointcloud_saver_node"), cloud_count_(0), tf_buffer(this->get_clock()), tf_listener(tf_buffer) {
            sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/camera/depth/color/points", 10, 
                std::bind(&PointCloudSaver::cloudCallback, this, std::placeholders::_1));
        }

    private:
        void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
            PCLcloud::Ptr cloud(new PCLcloud);
            pcl::fromROSMsg(*msg, *cloud);

            // Transform the point cloud to base_link frame
            if (!transformPointCloudToBaseLink(cloud, msg->header.frame_id))
                return;

            // Save the transformed point cloud to a file
            std::string filename = "cloud_" + std::to_string(cloud_count_) + ".pcd";
            pcl::io::savePCDFileASCII(filename, *cloud);
            RCLCPP_INFO(this->get_logger(), "Saved transformed point cloud to %s", filename.c_str());

            cloud_count_++;
        }

        bool transformPointCloudToBaseLink(PCLcloud::Ptr &cloud, const std::string &frame_id)
        {
            geometry_msgs::msg::TransformStamped transform_stamped;
            try
            {
                transform_stamped = tf_buffer.lookupTransform("base_link", frame_id, tf2::TimePointZero);
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Could not transform %s to base_link: %s", frame_id.c_str(), ex.what());
                return false;
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
            pcl::transformPointCloud(*cloud, *cloud, transform);
            return true;
        }

                rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
                tf2_ros::Buffer tf_buffer;
                tf2_ros::TransformListener tf_listener;
                int cloud_count_;
        };

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudSaver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
               