#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_search.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <thread>
#include <iostream>
#include <string>

using namespace std::placeholders;
using PointT = pcl::PointXYZRGB;
using PCLcloud = pcl::PointCloud<PointT>;

class PCLConverter : public rclcpp::Node
{
public:
    PCLConverter() : Node("pcl_converter_node"), tf_buffer(this->get_clock()), tf_listener(tf_buffer)
    {
        sub_pointcloud = create_subscription<sensor_msgs::msg::PointCloud2>("/camera/depth/color/points", 10, std::bind(&PCLConverter::cloudProcessing, this, _1));
        pub_process_cloud = create_publisher<sensor_msgs::msg::PointCloud2>("processed_points", 10);
        accumulated_cloud.reset(new PCLcloud);
        
        input_thread = std::thread(&PCLConverter::keyboardListener, this);
    }

    ~PCLConverter()
    {
        if (input_thread.joinable())
        {
            input_thread.join();
        }
    }

private:
    void cloudProcessing(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        PCLcloud::Ptr current_cloud(new PCLcloud);
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *current_cloud);

        if (!transformPointCloudToBaseLink(current_cloud, msg->header.frame_id))
            return;

        applyFilters(current_cloud, filtered_cloud);

        downSample(filtered_cloud, accumulated_cloud);

        publishProcessedCloud(accumulated_cloud);
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

    void applyFilters(const PCLcloud::Ptr &input_cloud, PCLcloud::Ptr &output_cloud)
    {
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(input_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, 3.0);
        pass.filter(*output_cloud);

        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(output_cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter(*output_cloud);

        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(output_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*output_cloud);

        pcl::SACSegmentation<PointT> seg;
        pcl::ExtractIndices<PointT> extract;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true); //True =    False =
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.001);

        seg.setInputCloud(output_cloud);
        seg.segment(*inliers, *coefficients);

        extract.setInputCloud(output_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); //True =    False =
        extract.filter(*output_cloud);
    }

    void downSample(const PCLcloud::Ptr &input_cloud, PCLcloud::Ptr &output_cloud)
    {
        pcl::octree::OctreePointCloudSearch<PointT> octree(0.01f);
        octree.setInputCloud(input_cloud);
        octree.addPointsFromInputCloud();

        for (auto it = input_cloud->begin(); it != input_cloud->end(); ++it)
        {
            PointT point = *it;
            std::vector<int> point_idx_vec;
            if (octree.voxelSearch(point, point_idx_vec))
            {
                pcl::PointCloud<PointT>::Ptr voxel_cloud(new pcl::PointCloud<PointT>);
                for (int idx : point_idx_vec)
                {
                    voxel_cloud->push_back((*input_cloud)[idx]);
                }

                PointT centroid;
                pcl::computeCentroid(*voxel_cloud, centroid);
                output_cloud->push_back(centroid);
            }
        }
    }

    void publishProcessedCloud(const PCLcloud::Ptr &cloud)
    {
        sensor_msgs::msg::PointCloud2 ros2_cloud;
        pcl::toROSMsg(*cloud, ros2_cloud);
        ros2_cloud.header.frame_id = "base_link";
        ros2_cloud.header.stamp = this->get_clock()->now();
        pub_process_cloud->publish(ros2_cloud);
    }

    void savePointCloud()
    {
        PCLcloud::Ptr cloud_without_camera(new PCLcloud);
        *cloud_without_camera = *accumulated_cloud;

        std::string filename = "mapped_cloud.ply";
        pcl::io::savePLYFile(filename, *cloud_without_camera);
        RCLCPP_INFO(this->get_logger(), "Saved point cloud to %s", filename.c_str());
    }

    void keyboardListener()
    {
        std::string input;
        while (rclcpp::ok())
        {
            std::getline(std::cin, input);
            if (input == "save")
            {
                savePointCloud();
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_process_cloud;
    PCLcloud::Ptr accumulated_cloud;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    std::thread input_thread;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
