//cpp
#include <iostream>
#include <Eigen/Dense>


//ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

//PCL
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_search.h>

//Alias
using namespace std::placeholders;
using Pointcloud2 = sensor_msgs::msg::PointCloud2;
using PointT = pcl::PointXYZ;
using PCLcloud = pcl::PointCloud<PointT>;


class OutlierRemoval : public rclcpp::Node {

    OutlierRemoval () : Node ("OutlierRemoval"), tf_buffer(this->get_clock()), tf_listener(tf_buffer){ 
        cloud_sub = create_subscription<Pointcloud2>("/camera/depth/color/points", 10, std::bind(&OutlierRemoval::cloudProcessing, this, _1));
        filtered_cloud_pub = create_publisher<Pointcloud2>("OutlierRemoval", 10);
    
    }
    

    private:
        void cloudProcessing (const Pointcloud2::SharedPtr msg) {
            PCLcloud::Ptr cur_cloud(new PCLcloud);
            PCLcloud::Ptr filter_cloud(new PCLcloud);
            // to pcl from ros2 msg
            pcl::fromROSMsg(*msg, *cur_cloud);

            //Transform cloud to base link frame 
            if (!transformPointCloudToBaseLink(cur_cloud, msg->header.frame_id))
            return;

            //Apply filter
            applyFilter(cur_cloud, filter_cloud);

            //Publish filtered 
            publishFilteredCloud(filter_cloud);

        }
        //Statiscal outlier 
        void applyFilter(const PCLcloud::Ptr &cur_cloud, PCLcloud::Ptr &filter_cloud) {
            // Initialize the output cloud
            *filter_cloud = *cur_cloud;

            // Statistical outlier removal for noise reduction
            pcl::StatisticalOutlierRemoval<PointT> sor;
            sor.setInputCloud(filter_cloud);
            sor.setMeanK(50);
            sor.setStddevMulThresh(1.0);
            sor.filter(*filter_cloud);

        }

        void publishFilteredCloud(const PCLcloud::Ptr &cloud) {
            // Create a new PointCloud2 message to hold the colored point cloud
            Pointcloud2 ros2_cloud;
            pcl::toROSMsg(*cloud, ros2_cloud);

            // Set the frame ID and timestamp
            ros2_cloud.header.frame_id = "base_link";
            ros2_cloud.header.stamp = this->get_clock()->now();

            // Compute the size of each point in bytes
            unsigned int point_size = ros2_cloud.point_step;
            unsigned int rgb_offset = 16; // Offset to RGB data in the binary representation

            // Iterate over each point in the point cloud and set its color to green (RGB: 0, 255, 0)
            for (size_t i = 0; i < ros2_cloud.width * ros2_cloud.height; ++i) {
                // Compute the start of the RGB data for this point
                unsigned int offset = i * point_size + rgb_offset;

                // Set RGB values directly in the byte array
                ros2_cloud.data[offset] = 255;   // R
                ros2_cloud.data[offset + 1] = 0; // G
                ros2_cloud.data[offset + 2] = 0;   // B
            }

            // Publish the modified point cloud
            filtered_cloud_pub->publish(ros2_cloud);
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
        
        rclcpp::Publisher<Pointcloud2>:: SharedPtr filtered_cloud_pub;
        rclcpp::Subscription<Pointcloud2>::SharedPtr  cloud_sub;
        tf2_ros::Buffer tf_buffer;
        tf2_ros::TransformListener tf_listener;


};

int main (int argc, char** argv) {

    rclcpp::init(argc, argv);
    auto node = std::make_shared<OutlierRemoval>();
    rclcpp::spin(node);
    return 0;
}