#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <vector>
#include <string>
#include <filesystem>
#include <memory>

using PointT = pcl::PointXYZRGB;
using PCLCloud = pcl::PointCloud<PointT>;

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
        PCLCloud::Ptr combined_cloud(new PCLCloud);
        std::vector<std::string> filenames;
        int batch_size = 100;  // Process in batches of 100 files

        for (int i = 0; i < 1360; i += batch_size)
        {
            PCLCloud::Ptr batch_cloud(new PCLCloud);

            for (int j = 0; j < batch_size; ++j)
            {
                int index = i + j;
                std::string filename = "cloud_" + std::to_string(index) + ".pcd";

                PCLCloud::Ptr cloud(new PCLCloud);
                if (pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
                {
                    RCLCPP_WARN(this->get_logger(), "No more files to load, stopping at %s", filename.c_str());
                    break;
                }

                if (!cloud->empty())
                {
                    *batch_cloud += *cloud;
                    filenames.push_back(filename);
                    RCLCPP_INFO(this->get_logger(), "Successfully combined %s", filename.c_str());
                }
            }

            // Apply filtering and downsampling on the batch
            PCLCloud::Ptr filtered_cloud = applyStatisticalOutlierRemoval(batch_cloud);
            PCLCloud::Ptr voxel_filtered_cloud = applyVoxelGridFilter(filtered_cloud);

            // Combine with the main point cloud
            *combined_cloud += *voxel_filtered_cloud;

            RCLCPP_INFO(this->get_logger(), "Processed batch %d to %d", i, i + batch_size - 1);
        }

        // Apply final smoothing and outlier removal
        PCLCloud::Ptr smoothed_cloud = applyMovingLeastSquares(combined_cloud);
        PCLCloud::Ptr cleaned_cloud = applyRadiusOutlierRemoval(smoothed_cloud);
        savePointCloud(cleaned_cloud, "combined_cloud.pcd");
        deletePointCloudFiles(filenames);
    }

    PCLCloud::Ptr applyStatisticalOutlierRemoval(const PCLCloud::Ptr& input_cloud)
    {
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);

        PCLCloud::Ptr filtered_cloud(new PCLCloud);
        sor.filter(*filtered_cloud);

        RCLCPP_INFO(this->get_logger(), "Applied Statistical Outlier Removal");
        return filtered_cloud;
    }

    PCLCloud::Ptr applyVoxelGridFilter(const PCLCloud::Ptr& input_cloud)
    {
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(input_cloud);
        voxel_grid.setLeafSize(0.005f, 0.005f, 0.005f);

        PCLCloud::Ptr voxel_filtered_cloud(new PCLCloud);
        voxel_grid.filter(*voxel_filtered_cloud);

        RCLCPP_INFO(this->get_logger(), "Applied Voxel Grid Downsampling");
        return voxel_filtered_cloud;
    }

    PCLCloud::Ptr applyMovingLeastSquares(const PCLCloud::Ptr& input_cloud)
    {
        pcl::MovingLeastSquares<PointT, PointT> mls;
        mls.setInputCloud(input_cloud);
        mls.setSearchRadius(0.02);
        mls.setPolynomialOrder(2);
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<PointT, PointT>::NONE);

        PCLCloud::Ptr smoothed_cloud(new PCLCloud);
        mls.process(*smoothed_cloud);

        RCLCPP_INFO(this->get_logger(), "Applied Moving Least Squares Smoothing");
        return smoothed_cloud;
    }

    PCLCloud::Ptr applyRadiusOutlierRemoval(const PCLCloud::Ptr& input_cloud)
    {
        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setInputCloud(input_cloud);
        ror.setRadiusSearch(0.02);
        ror.setMinNeighborsInRadius(5);

        PCLCloud::Ptr cleaned_cloud(new PCLCloud);
        ror.filter(*cleaned_cloud);

        RCLCPP_INFO(this->get_logger(), "Applied Radius Outlier Removal");
        return cleaned_cloud;
    }

    void savePointCloud(const PCLCloud::Ptr& cloud, const std::string& filename)
    {
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        pcl::io::savePCDFileASCII(filename, *cloud);
        RCLCPP_INFO(this->get_logger(), "Saved combined point cloud to %s", filename.c_str());
    }

    void deletePointCloudFiles(const std::vector<std::string>& filenames)
    {
        for (const auto& file : filenames)
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

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudCombiner>());
    rclcpp::shutdown();
    return 0;
}
