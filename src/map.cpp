#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud.h>

// Alias
using PointT = pcl::PointXYZRGB;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PCLcloud = pcl::PointCloud<PointT>;
using Octree = pcl::octree::OctreePointCloud<PointT>;
using Voxel = pcl::VoxelGrid<PointT>; 

class PCLConverter : public rclcpp::Node
{
public:
    PCLConverter() : Node("pcl_converter_node"), octree(0.01f)
    {
        // Create a subscription to the PointCloud2 topic
        sub_pointcloud = create_subscription<PointCloud2>("/camera/depth/color/points", 10, std::bind(&PCLConverter::cloudProcessing, this, std::placeholders::_1));
        pub_process_cloud = create_publisher<PointCloud2>("processed_points", 10);
        accumulated_cloud.reset(new PCLcloud);
    }

private:
    void cloudProcessing(const PointCloud2::SharedPtr msg)
    {
        // Convert the ROS PointCloud2 message to a PCL point cloud
        PCLcloud::Ptr current_cloud(new PCLcloud);
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *current_cloud);

        // Process the point cloud here
        downSample(current_cloud, filtered_cloud);
        addOctree(filtered_cloud);

        // Processed pointcloud to ros2 msg
        PointCloud2 ros2_cloud;
        toROS2(accumulated_cloud, ros2_cloud);
        pub_process_cloud->publish(ros2_cloud);
    }

    void toROS2(const PCLcloud::Ptr &process_pcl_cloud, PointCloud2 &ros2_cloud) {
        pcl::toROSMsg(*process_pcl_cloud, ros2_cloud);
        ros2_cloud.header.frame_id = "base_link"; // Set the frame ID
        ros2_cloud.header.stamp = this->get_clock()->now();
    }

    void downSample(const PCLcloud::Ptr &current_cloud, const PCLcloud::Ptr &filtered_cloud) {
        auto leaf_size = 0.01f;
        Voxel sor;
        sor.setInputCloud(current_cloud);
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*filtered_cloud);
    }

    void addOctree(const PCLcloud::Ptr &filtered_cloud) { 
        octree.setInputCloud(filtered_cloud);
        octree.addPointsFromInputCloud();
        *accumulated_cloud += *filtered_cloud;
        
        std::vector<PointT, Eigen::aligned_allocator<PointT>> voxel_centers;
        octree.getOccupiedVoxelCenters(voxel_centers);

        accumulated_cloud->clear();
        for (const auto& point : voxel_centers) {
            accumulated_cloud->push_back(point);
        }
    }

    rclcpp::Subscription<PointCloud2>::SharedPtr sub_pointcloud;
    rclcpp::Publisher<PointCloud2>::SharedPtr pub_process_cloud;
    PCLcloud::Ptr accumulated_cloud;
    Octree octree; 
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
