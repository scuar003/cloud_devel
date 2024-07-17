#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/octree/octree_search.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <thread>
#include <ncurses.h> // For key press event handling

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
        pub_mesh_cloud = create_publisher<PointCloud2>("mesh_points", 10);
        accumulated_cloud.reset(new PCLcloud);
    }

    void start()
    {
        std::thread key_thread(&PCLConverter::keyPressListener, this);
        key_thread.detach();
        rclcpp::spin(this->shared_from_this());
    }

private:
    void cloudProcessing(const PointCloud2::SharedPtr msg)
    {
        if (stop_mapping) return;

        PCLcloud::Ptr current_cloud(new PCLcloud);
        PCLcloud::Ptr filtered_cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *current_cloud);

        // Transform the point cloud to the base_link frame
        if (!transformPointCloudToBaseLink(current_cloud, msg->header.frame_id))
            return;

        // Filter the point cloud
        applyFilters(current_cloud, filtered_cloud);

        // Downsample and accumulate the point cloud
        downSample(filtered_cloud, accumulated_cloud);

        // Publish the processed point cloud
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
        // Voxel grid filter for downsampling
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(input_cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter(*output_cloud);

        // Statistical outlier removal for noise reduction
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(output_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*output_cloud);

        // Segmentation to remove planes (ground)
        pcl::SACSegmentation<PointT> seg;
        pcl::ExtractIndices<PointT> extract;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.001);

        seg.setInputCloud(output_cloud);
        seg.segment(*inliers, *coefficients);

        extract.setInputCloud(output_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); // remove the plane
        extract.filter(*output_cloud);
    }

    void downSample(const PCLcloud::Ptr &input_cloud, PCLcloud::Ptr &output_cloud)
    {
        pcl::octree::OctreePointCloudSearch<PointT> octree(0.01f); // Adjust resolution as needed
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
        PointCloud2 ros2_cloud;
        pcl::toROSMsg(*cloud, ros2_cloud);
        ros2_cloud.header.frame_id = "base_link";
        ros2_cloud.header.stamp = this->get_clock()->now();
        pub_process_cloud->publish(ros2_cloud);
    }

    void keyPressListener()
    {
        initscr();
        timeout(-1); // Wait indefinitely for a key press

        while (true)
        {
            int ch = getch();
            if (ch == 'c')
            {
                stop_mapping = true;
                createMesh(accumulated_cloud);
                break;
            }
        }

        endwin();
    }

    void createMesh(const PCLcloud::Ptr &cloud)
    {
        // Estimate normals
        pcl::NormalEstimation<PointT, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

        tree->setInputCloud(cloud);
        n.setInputCloud(cloud);
        n.setSearchMethod(tree);
        n.setKSearch(50);  // Increased number of neighbors for normal estimation
        n.compute(*normals);

        // Concatenate the XYZ and normal fields
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

        // Create search tree
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        tree2->setInputCloud(cloud_with_normals);

        // Initialize objects
        pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
        pcl::PolygonMesh triangles;

        // Set parameters for the triangulation
        gp3.setSearchRadius(0.1);  // Increased search radius
        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(300);  // Further increased maximum nearest neighbors
        gp3.setMaximumSurfaceAngle(M_PI / 2);  // 90 degrees
        gp3.setMinimumAngle(M_PI / 36);  // 5 degrees
        gp3.setMaximumAngle(M_PI / 2);  // 90 degrees
        gp3.setNormalConsistency(false);

        // Perform reconstruction
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree2);
        gp3.reconstruct(triangles);

        // Convert the mesh to a point cloud
        PCLcloud::Ptr mesh_cloud(new PCLcloud);
        pcl::fromPCLPointCloud2(triangles.cloud, *mesh_cloud);

        // Publish the mesh as a point cloud
        PointCloud2 ros2_mesh_cloud;
        pcl::toROSMsg(*mesh_cloud, ros2_mesh_cloud);
        ros2_mesh_cloud.header.frame_id = "base_link";
        ros2_mesh_cloud.header.stamp = this->get_clock()->now();
        pub_mesh_cloud->publish(ros2_mesh_cloud);

        RCLCPP_INFO(this->get_logger(), "Mesh published to topic 'mesh_points'");
    }

    rclcpp::Subscription<PointCloud2>::SharedPtr sub_pointcloud;
    rclcpp::Publisher<PointCloud2>::SharedPtr pub_process_cloud;
    rclcpp::Publisher<PointCloud2>::SharedPtr pub_mesh_cloud;
    PCLcloud::Ptr accumulated_cloud;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    bool stop_mapping = false;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLConverter>();
    node->start();
    rclcpp::shutdown();
    return 0;
}