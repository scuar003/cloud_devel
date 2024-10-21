#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <string>
#include <memory>

using PointT = pcl::PointXYZRGB;
using PCLCloud = pcl::PointCloud<PointT>;

class MeshImprovementNode : public rclcpp::Node
{
public:
    MeshImprovementNode() : Node("mesh_improvement_node")
    {
        improveMeshQuality();
    }

private:
    void improveMeshQuality()
    {
        // Load point cloud from file
        PCLCloud::Ptr cloud(new PCLCloud);
        if (pcl::io::loadPCDFile<PointT>("combined_cloud.pcd", *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load point cloud file.");
            return;
        }

        // Downsample and densify the point cloud using a Voxel Grid filter
        PCLCloud::Ptr cloud_filtered = applyVoxelGridFilter(cloud, 0.002f); // Moderate leaf size for better balance between density and computation

        // Remove noise using Statistical Outlier Removal filter
        PCLCloud::Ptr cloud_denoised = applyStatisticalOutlierRemoval(cloud_filtered, 100, 1.0); // Fine-tuned parameters for noise removal without losing valuable data

        if (cloud_denoised->empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Filtered point cloud is empty after applying filters.");
            return;
        }

        // Smooth the point cloud using Moving Least Squares (MLS)
        PCLCloud::Ptr cloud_smoothed = applyMovingLeastSquares(cloud_denoised, 0.01); // Reduced search radius for more detail preservation

        // Estimate normals for the point cloud
        pcl::PointCloud<pcl::Normal>::Ptr normals = estimateNormals(cloud_smoothed, 30); // Adjusted KSearch for more accurate normal estimation

        // Concatenate XYZRGB and normal fields
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields(*cloud_smoothed, *normals, *cloud_with_normals);

        // Perform mesh reconstruction using Greedy Projection Triangulation to retain color information
        pcl::PolygonMesh mesh = performGreedyProjectionTriangulation(cloud_with_normals);

        // Save the final improved mesh
        if (!mesh.polygons.empty())
        {
            pcl::io::savePLYFile("improved_mesh.ply", mesh);
            RCLCPP_INFO(this->get_logger(), "Saved improved mesh to improved_mesh.ply");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Mesh reconstruction failed: no data to save.");
        }
    }

    PCLCloud::Ptr applyVoxelGridFilter(const PCLCloud::Ptr &cloud, float leaf_size)
    {
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
        PCLCloud::Ptr cloud_filtered(new PCLCloud);
        voxel_grid.filter(*cloud_filtered);
        return cloud_filtered;
    }

    PCLCloud::Ptr applyStatisticalOutlierRemoval(const PCLCloud::Ptr &cloud, int mean_k, double stddev_mul_thresh)
    {
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(mean_k);
        sor.setStddevMulThresh(stddev_mul_thresh);
        PCLCloud::Ptr cloud_denoised(new PCLCloud);
        sor.filter(*cloud_denoised);
        return cloud_denoised;
    }

    PCLCloud::Ptr applyMovingLeastSquares(const PCLCloud::Ptr &cloud, double search_radius)
    {
        pcl::MovingLeastSquares<PointT, PointT> mls;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        mls.setInputCloud(cloud);
        mls.setPolynomialOrder(2);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(search_radius);
        mls.setComputeNormals(true);
        PCLCloud::Ptr cloud_smoothed(new PCLCloud);
        mls.process(*cloud_smoothed);
        return cloud_smoothed;
    }

    pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(const PCLCloud::Ptr &cloud, int k_search)
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        ne.setSearchMethod(tree);
        ne.setInputCloud(cloud);
        ne.setKSearch(k_search);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.compute(*normals);
        return normals;
    }

    pcl::PolygonMesh performGreedyProjectionTriangulation(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud_with_normals)
    {
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        kdtree->setInputCloud(cloud_with_normals);

        pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
        pcl::PolygonMesh mesh;

        gp3.setSearchRadius(0.02);
        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(100);
        gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
        gp3.setMinimumAngle(M_PI / 18);        // 10 degrees
        gp3.setMaximumAngle(2 * M_PI / 3);     // 120 degrees
        gp3.setNormalConsistency(false);

        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(kdtree);
        gp3.reconstruct(mesh);

        return mesh;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MeshImprovementNode>());
    rclcpp::shutdown();
    return 0;
}
