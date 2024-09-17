#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <vector>
#include <string>
#include <filesystem>

using PointT = pcl::PointXYZRGB;
using PCLcloud = pcl::PointCloud<PointT>;

class PointCloudCombiner : public rclcpp::Node
{
public:
    PointCloudCombiner() : Node("pointcloud_combiner_node")
    {
        combinePointClouds();
    }

private:
    void combinePointClouds()
    {
        PCLcloud::Ptr combined_cloud(new PCLcloud);
        int i = 0;
        std::vector<std::string> filenames;

        // Load multiple point clouds from files and combine them
        while (true)
        {
            PCLcloud::Ptr cloud(new PCLcloud);
            std::string filename = "cloud_" + std::to_string(i) + ".pcd";

            // Try to load the point cloud file
            if (pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
            {
                // If the file doesn't exist, break the loop
                RCLCPP_WARN(this->get_logger(), "No more files to load, stopping at %s", filename.c_str());
                break;
            }

            // Check if the cloud is not empty before adding
            if (!cloud->empty())
            {
                *combined_cloud += *cloud;     // Combine the point clouds
                filenames.push_back(filename); // Store the filename for later deletion
                RCLCPP_INFO(this->get_logger(), "Successfully combined %s", filename.c_str());
            }

            // Move to the next file
            i++;
        }

        // Apply filters and processing on the combined point cloud

        // 1. Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(combined_cloud);
        sor.setMeanK(50);            // 50 nearest neighbors for calculating mean distance
        sor.setStddevMulThresh(1.0); // Points further than 1 standard deviation from the mean will be removed
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        sor.filter(*filtered_cloud);

        RCLCPP_INFO(this->get_logger(), "Applied Statistical Outlier Removal");

        // 2. Plane Segmentation (removing large planar surfaces)
        pcl::SACSegmentation<PointT> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        PCLcloud::Ptr cloud_no_plane(new PCLcloud);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.01); // Max distance to the model plane

        // Segment the largest planar component from the cloud
        seg.setInputCloud(filtered_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            RCLCPP_WARN(this->get_logger(), "No planar surface found in the combined point cloud.");
        }
        else
        {
            // Extract the inliers (i.e., remove the plane)
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(filtered_cloud);
            extract.setIndices(inliers);
            extract.setNegative(true); // True to extract everything except the plane
            extract.filter(*cloud_no_plane);

            RCLCPP_INFO(this->get_logger(), "Plane segmentation complete. Extracted %ld points.", cloud_no_plane->points.size());
        }

        // 3. Octree Downsampling with Color Preservation
        pcl::octree::OctreePointCloudPointVector<PointT> octree(0.01f); // 1 cm resolution
        octree.setInputCloud(cloud_no_plane);
        octree.addPointsFromInputCloud();

        PCLcloud::Ptr octree_filtered_cloud(new PCLcloud);

        // Iterate through all leaf nodes in the octree
        for (auto it = octree.begin(); it != octree.end(); ++it)
        {
            // Get all points in the current leaf node
            const std::vector<int> &indices = it.getLeafContainer().getPointIndicesVector();

            Eigen::Vector3f color_avg(0.0, 0.0, 0.0);
            Eigen::Vector3f point_avg(0.0, 0.0, 0.0);

            for (int idx : indices)
            {
                const PointT &point = (*cloud_no_plane)[idx];
                point_avg += point.getVector3fMap();
                color_avg += Eigen::Vector3f(point.r, point.g, point.b);
            }

            point_avg /= indices.size();
            color_avg /= indices.size();

            PointT avg_point;
            avg_point.x = point_avg.x();
            avg_point.y = point_avg.y();
            avg_point.z = point_avg.z();
            avg_point.r = static_cast<uint8_t>(color_avg.x());
            avg_point.g = static_cast<uint8_t>(color_avg.y());
            avg_point.b = static_cast<uint8_t>(color_avg.z());

            octree_filtered_cloud->points.push_back(avg_point);
        }

        RCLCPP_INFO(this->get_logger(), "Applied Octree downsampling with color preservation");

        // 4. Set width and height before saving the final cloud
        octree_filtered_cloud->width = octree_filtered_cloud->points.size();
        octree_filtered_cloud->height = 1; // Unorganized point cloud
        octree_filtered_cloud->is_dense = true;

        // Save the combined and processed point cloud
        pcl::io::savePCDFileASCII("combined_cloud.pcd", *octree_filtered_cloud);
        RCLCPP_INFO(this->get_logger(), "Saved combined point cloud to combined_cloud.pcd");

        // Delete the individual cloud files after combining
        for (const auto &file : filenames)
        {
            if (std::filesystem::remove(file))
            {
                RCLCPP_INFO(this->get_logger(), "Deleted file %s", file.c_str());
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Failed to delete file %s", file.c_str());
            }
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudCombiner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
