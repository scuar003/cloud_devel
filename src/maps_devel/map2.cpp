#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/common/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>

#include <tuple>

template<typename P>
std::tuple<uint8_t, uint8_t, uint8_t> fromRGB(P &point) {
    // unpack rgb into r/g/b
    uint32_t rgb = (uint32_t)(point.rgb);
    uint8_t r = (rgb >> 16) & 0x0000ff;
    uint8_t g = (rgb >> 8)  & 0x0000ff;
    uint8_t b = (rgb)       & 0x0000ff;
    return std::make_tuple(r,g,b);
}

/*
template<typename P>
std::tuple<float, float, float> fromRGB(const P &point) {
    // unpack rgb into r/g/b
    float *rgb = (float *)(&point.rgb);
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];
    return std::make_tuple(r,g,b);
}
*/

// Alias
using PointT = pcl::PointXYZRGB;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PCLcloud = pcl::PointCloud<PointT>;
using Octree = pcl::octree::OctreePointCloud<PointT>;
using Voxel = pcl::VoxelGrid<PointT>;

class PCLConverter : public rclcpp::Node
{
public:
    PCLConverter() : Node("pcl_converter_node"), octree(0.01f), tf_buffer(this->get_clock()), tf_listener(tf_buffer)
    {
        sub_pointcloud = create_subscription<PointCloud2>("/camera/depth/color/points", 10, std::bind(&PCLConverter::cloudProcessing, this, std::placeholders::_1));
        pub_process_cloud = create_publisher<PointCloud2>("processed_points", 10);
        accumulated_cloud.reset(new PCLcloud);
    }

private:
    void cloudProcessing(const PointCloud2::SharedPtr msg)
    {
        PCLcloud::Ptr current_cloud(new PCLcloud);
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *current_cloud);

        // for ( const auto &field : msg->fields) 
        //     std::cout << field.name << " (" << field.offset << ") " << field.count << " [" << ((int)field.datatype) << "] ";
        // std::cout << std::endl;

        // for (const auto & point : *current_cloud) {
        //     std::cout << point.x << ',' << point.y << ',' << point.z;
        //     auto [r,g,b] = fromRGB(point);
        //     std::cout << " ["  << r << ',' << g << ',' << b << ']' << "]" << std::endl;
        // }
        
        

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

        downSample(current_cloud, filtered_cloud);
        addOctree(filtered_cloud);

        PointCloud2 ros2_cloud;
        toROS2(filtered_cloud, ros2_cloud);
        pub_process_cloud->publish(ros2_cloud);
        
    }

    void toROS2(const PCLcloud::Ptr &process_pcl_cloud, PointCloud2 &ros2_cloud) {
        pcl::toROSMsg(*process_pcl_cloud, ros2_cloud);
        ros2_cloud.header.frame_id = "base_link";
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
