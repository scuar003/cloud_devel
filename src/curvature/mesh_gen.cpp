#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <random>

using namespace std::placeholders;
using PointT = pcl::PointXYZRGB;


class TerrainPlaneSegmentation : public rclcpp::Node {
public:
    TerrainPlaneSegmentation() : Node("terrain_plane_segmentation") {
        // Subscriber for point cloud data
        cloud_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/depth/color/points", 10, std::bind(&TerrainPlaneSegmentation::cloudMsg, this, _1));
        
        cmd_sub_ = create_subscription <std_msgs::msg::String> (
            "menu_action", 10, std::bind(&TerrainPlaneSegmentation::cmdCallback, this, _1));

        // Publishers for segmented planes and curvature
        plane_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_planes", 10);
        curvature_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/curvature_visualization", 10);

        RCLCPP_INFO(this->get_logger(), "TerrainPlaneSegmentation node has been started.");
    }

private:

    void cloudMsg(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        cloud_msg = msg;
    }

    void cmdCallback (const std_msgs::msg::String cmd) {
        if (cmd.data == "detect surfaces")
        RCLCPP_INFO(get_logger(), "Detecting surfaces...");
        pointCloudCallback ();

    }


    void pointCloudCallback() {
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*cloud_msg, *cloud);
        
        RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points.", cloud->points.size());

        // Process the point cloud to segment planes and curvature
        segmentPlanesAndCurvature(cloud, cloud_msg->header);
    }

    void segmentPlanesAndCurvature(const pcl::PointCloud<PointT>::Ptr &cloud, const std_msgs::msg::Header &header) {
        
        pcl::ExtractIndices<PointT> extract;
        pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>(*cloud));
        pcl::PointCloud<PointT>::Ptr all_planes_cloud(new pcl::PointCloud<PointT>);

        pcl::VoxelGrid<PointT> voxel_grid;
        auto leaf_size = 0.0001f;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_grid.filter(*cloud_filtered);
    
        
        // 1. Segment planes using RANSAC
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.015); // Adjust threshold based on terrain scale

        

        int num_planes = 0;

        // Random color generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> color_dist(0, 255);

        while (cloud_filtered->points.size() > 100) { // Minimum points for processing
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.empty()) {
                RCLCPP_INFO(this->get_logger(), "No more planes found.");
                break;
            }

            // Extract plane points
            pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>);
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane_cloud);

            // Assign a random color to this plane
            uint8_t r = color_dist(gen);
            uint8_t g = color_dist(gen);
            uint8_t b = color_dist(gen);

            for (auto &point : plane_cloud->points) {
                point.r = r;
                point.g = g;
                point.b = b;
            }

            *all_planes_cloud += *plane_cloud;  // Collect all planes for visualization

            RCLCPP_INFO(this->get_logger(), "Plane %d segmented with %zu points.", ++num_planes, plane_cloud->points.size());

            // Extract remaining points
            extract.setNegative(true);
            pcl::PointCloud<PointT> remaining_points;
            extract.filter(remaining_points);
            *cloud_filtered = remaining_points;
        }

        // Publish segmented planes with different colors
        if (!all_planes_cloud->points.empty()) {
            sensor_msgs::msg::PointCloud2 planes_msg;
            pcl::toROSMsg(*all_planes_cloud, planes_msg);
            planes_msg.header = header;  // Use the same header as the input cloud
            plane_publisher_->publish(planes_msg);
            RCLCPP_INFO(this->get_logger(), "Published %zu points in segmented planes with colors.", all_planes_cloud->points.size());
        }

        // 2. Estimate normals and curvature with reduced radius for finer detail
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(0.0001); // Smaller radius for high-resolution curvature estimation
        ne.compute(*cloud_normals);

        // 3. Compute principal curvatures with the same reduced radius
        pcl::PrincipalCurvaturesEstimation<PointT, pcl::Normal, pcl::PrincipalCurvatures> pc;
        pc.setInputCloud(cloud);
        pc.setInputNormals(cloud_normals);
        pc.setSearchMethod(tree);
        pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr cloud_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
        pc.setRadiusSearch(0.0005); // Smaller radius for fine detail
        pc.compute(*cloud_curvatures);

        pcl::PointCloud<PointT>::Ptr curvature_visualization_cloud(new pcl::PointCloud<PointT>);
        for (size_t i = 0; i < cloud_curvatures->points.size(); ++i) {
            double curvature = cloud_curvatures->points[i].pc1; // Principal curvature
            PointT point = cloud->points[i];
            if (curvature > 0.005) {  // Lower threshold for smaller features
                curvature_visualization_cloud->push_back(point);  // Add points with noticeable curvature
            }
        }

        // Publish curvature visualization
        if (!curvature_visualization_cloud->points.empty()) {
            sensor_msgs::msg::PointCloud2 curvature_msg;
            pcl::toROSMsg(*curvature_visualization_cloud, curvature_msg);
            curvature_msg.header = header;  // Use the same header as the input cloud
            curvature_publisher_->publish(curvature_msg);
            RCLCPP_INFO(this->get_logger(), "Published %zu points in curvature visualization.", curvature_visualization_cloud->points.size());
        } else {
            RCLCPP_WARN(this->get_logger(), "No curvature points found to publish.");
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr curvature_publisher_;
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TerrainPlaneSegmentation>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
