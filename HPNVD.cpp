#define Pi 3.1415926
#define NULL_POINTID -1

#include <stdio.h>
#include <vector>
#include <time.h>
#include <pcl\point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/eigen.h>




typedef struct {
	float x;
	float y;
	float z;
}Vertex;

typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;

void HPVND_LRF_Z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex& z_axis)
{
	int i;
	pcl::PointXYZ query_point = cloud->points[0];
	// calculate covariance matrix
	Eigen::Matrix3f Cov;
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud, centroid);
	pcl::computeCovarianceMatrix(*cloud, centroid, Cov);
	EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_min;
	EIGEN_ALIGN16 Eigen::Vector3f normal;
	pcl::eigen33(Cov, eigen_min, normal);
	z_axis.x = normal(0);
	z_axis.y = normal(1);
	z_axis.z = normal(2);
	// z-axis sign disambiguity
	float z_sign = 0;
	for (i = 0; i < cloud->points.size(); i++)
	{
		float vec_x = query_point.x - cloud->points[i].x;
		float vec_y = query_point.y - cloud->points[i].y;
		float vec_z = query_point.z - cloud->points[i].z;
		z_sign += (vec_x * z_axis.x + vec_y * z_axis.y + vec_z * z_axis.z);
	}
	if (z_sign < 0)
	{
		z_axis.x = -z_axis.x;
		z_axis.y = -z_axis.y;
		z_axis.z = -z_axis.z;
	}
}
void HPVND_LRF_X_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, std::vector<float> PointDist, Vertex& x_axis)
{

	pcl::PointXYZ query_point = cloud->points[0];
	std::vector<std::vector<float>> bin_cloud_LDI;
	bin_cloud_LDI.resize(36);
	std::vector<std::vector<pcl::PointXYZ>> bin_point;
	bin_point.resize(36);
	for (int i = 0; i < cloud->points.size(); i++)
	{

		double a, b, c; // 法向量的分量
		double x, y, z; // 关键点p的坐标
		double x1, y1, z1; // 点（x1, y1, z1）的坐标
		a = z_axis.x;
		b = z_axis.y;
		c = z_axis.z;
		x = query_point.x;
		y = query_point.y;
		z = query_point.z;
		x1 = cloud->points[i].x;
		y1 = cloud->points[i].y;
		z1 = cloud->points[i].z;
		// 计算平面方程的常数项
		double d = -(a * x + b * y + c * z);
		// 求解投影点的坐标(qx,qy,qz)
		double t = (-a * x1 - b * y1 - c * z1 - d) / (a * a + b * b + c * c);
		double qx = x1 + a * t;
		double qy = y1 + b * t;
		double qz = z1 + c * t;

		//计算方位角
		double azimuthAngle = std::atan((qy - y) / (qx - x));
		//确定几号bin
		int bin_Index = azimuthAngle / (Pi / 18);
		//将该点放入bin内
		pcl::PointXYZ temp_point(qx, qy, qz);
		bin_point[bin_Index].push_back(temp_point);
		//计算该点的权重
		float LDI_temp = sup_radius - sqrt((qx - x) * (qx - x) + (qy - y) * (qy - y));
		LDI_temp = pow(LDI_temp, 2);
		bin_cloud_LDI[bin_Index].push_back(LDI_temp);

	}
	//找到权值最大的bin
	int binIndex;
	float max_temp = 0;
	for (int j = 0; j < 36; j++) {
		int sum = std::accumulate(bin_cloud_LDI[j].begin(), bin_cloud_LDI[j].end(), 0);
		if (sum > max_temp) {
			max_temp = sum;
			binIndex = j;
		}
	}
	//计算平均坐标
	std::vector<pcl::PointXYZ>& points = bin_point[binIndex];
	pcl::PointXYZ averagePoint(0, 0, 0);

	for (const pcl::PointXYZ& point : points) {
		averagePoint.x += point.x;
		averagePoint.y += point.y;
		averagePoint.z += point.z;
	}

	int numPoints = points.size();
	averagePoint.x /= numPoints;
	averagePoint.y /= numPoints;
	averagePoint.z /= numPoints;

	Vertex x_axis_temp = { 0.0f,0.0f,0.0f };
	x_axis_temp.x = averagePoint.x - query_point.x;
	x_axis_temp.y = averagePoint.y - query_point.y;
	x_axis_temp.z = averagePoint.z - query_point.z;
	//Normalization归一化
	float size = sqrt(pow(x_axis_temp.x, 2) + pow(x_axis_temp.y, 2) + pow(x_axis_temp.z, 2));
	x_axis_temp.x /= size;
	x_axis_temp.y /= size;
	x_axis_temp.z /= size;
	x_axis = x_axis_temp;

}
void HPVND_LRF_Y_axis(Vertex x_axis, Vertex z_axis, Vertex& y_axis)
{
	Eigen::Vector3f x(x_axis.x, x_axis.y, x_axis.z);
	Eigen::Vector3f z(z_axis.x, z_axis.y, z_axis.z);
	Eigen::Vector3f y;
	y = x.cross(z);//cross product
	y_axis.x = y(0);
	y_axis.y = y(1);
	y_axis.z = y(2);
}
void HPVND_LRF_for_cloud_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int> indices, float sup_radius, std::vector<LRF>& Cloud_LRF)
{
	int i, j, m;
	//
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>pointIdx;
	std::vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	//LRF calculation
	for (i = 0; i < indices.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);//local surface
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_z(new pcl::PointCloud<pcl::PointXYZ>);//local surface for computing the z-axis of LRF
		query_point = cloud->points[indices[i]];
		//
		if (kdtree.radiusSearch(query_point, sup_radius / 3, pointIdx, pointDst) > 3)
		{
			for (j = 0; j < pointIdx.size(); j++)
			{
				sphere_neighbor_z->points.push_back(cloud->points[pointIdx[j]]);
			}
			if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 10)//only if there are more than 10 points in the local surface
			{
				for (j = 0; j < pointIdx.size(); j++)
				{
					sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);
				}
				Vertex x_axis, y_axis, z_axis;
				HPVND_LRF_Z_axis(sphere_neighbor_z, z_axis);
				HPVND_LRF_X_axis(sphere_neighbor, z_axis, sup_radius, pointDst, x_axis);
				HPVND_LRF_Y_axis(x_axis, z_axis, y_axis);
				LRF temp = { indices[i],x_axis,y_axis,z_axis };
				Cloud_LRF.push_back(temp);
			}
			else
			{
				LRF temp = { NULL_POINTID,{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
				Cloud_LRF.push_back(temp);
			}
		}
		else
		{
			LRF temp = { NULL_POINTID,{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
			Cloud_LRF.push_back(temp);
		}
	}
}

void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, LRF pointLRF, pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_cloud)
{
	pcl::PointXYZ point = cloud->points[0];//the centroid of the local surface
	int number_of_points = cloud->points.size();
	transformed_cloud->points.resize(number_of_points);
	Eigen::Matrix3f matrix;
	matrix(0, 0) = pointLRF.x_axis.x; matrix(0, 1) = pointLRF.x_axis.y; matrix(0, 2) = pointLRF.x_axis.z;
	matrix(1, 0) = pointLRF.y_axis.x; matrix(1, 1) = pointLRF.y_axis.y; matrix(1, 2) = pointLRF.y_axis.z;
	matrix(2, 0) = pointLRF.z_axis.x; matrix(2, 1) = pointLRF.z_axis.y; matrix(2, 2) = pointLRF.z_axis.z;
	for (int i = 0; i < number_of_points; i++)
	{
		Eigen::Vector3f transformed_point(
			cloud->points[i].x - point.x,
			cloud->points[i].y - point.y,
			cloud->points[i].z - point.z);

		transformed_point = matrix * transformed_point;

		pcl::PointXYZ new_point;
		new_point.x = transformed_point(0);
		new_point.y = transformed_point(1);
		new_point.z = transformed_point(2);
		transformed_cloud->points[i] = new_point;
	}
}

void local_HPVND(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float sup_radius, int bin, std::vector<float>& histogram)
{
	int i;
	int number_of_points = cloud->points.size();
	std::vector<float>sub_histogram1, sub_histogram2, sub_histogram3;



	// 创建法线估计对象
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	// 创建一个Kd树对象，用于法线估计
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	ne.setSearchMethod(tree);
	// 设置法线估计的参数
	ne.setKSearch(10); // 设置K近邻搜索的数量
	// 创建一个空的法线点云
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// 计算法线
	ne.compute(*normals);

	//View1,投影到xy平面
	std::vector<std::vector<float>> bin_cloud_LDI1;
	bin_cloud_LDI1.resize(108);
	for (int i = 0; i < number_of_points; i++)
	{
		//calculate bin's index
		//将投影坐标转换为极坐标，确定它在哪个圆圈中
		float rou = std::sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y);
		double theta = std::atan2(cloud->points[i].y, cloud->points[i].x);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta = std::fmod(theta + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta < 0) {
			theta += 2 * Pi;
		}
		// 将角度映射到 8 个部分
		int section = static_cast<int>(theta / (Pi / 4)) + 1;
		if (section > 8) {
			section = 1;
		}
		int section2 = rou / (sup_radius / 5);
		section2 += 1;
		if (section2 == 3)
			section2 = 2;
		else if (section2 == 4 || section2 == 5)
			section2 = 3;


		//计算法向量的贡献方向
		int section3;
		double minDiff = std::numeric_limits<double>::max();//法向量与贡献方向的夹角

		//计算投影法向量的方向
		double theta2 = std::atan2(normals->points[i].normal_y, normals->points[i].normal_x);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta2 = std::fmod(theta2 + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta2 < 0) {
			theta2 += 2 * Pi;
		}

		if (section2 == 3) {
			// 计算与四个方向之间的差值
			double directions[4] = { 0, M_PI / 2, M_PI, 3 * M_PI / 2 };
			for (int i = 0; i < 4; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else if (section2 == 2)
		{
			// 计算与八个方向之间的差值
			double directions[8] = { 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI, 5 * M_PI / 4, 3 * M_PI / 2, 7 * M_PI / 4 };
			for (int i = 0; i < 8; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else {
			// 计算与十二个方向之间的差值
			double directions[12] = { 0, M_PI / 6, M_PI / 3, M_PI / 2, 2 * M_PI / 3, 5 * M_PI / 6, M_PI, 7 * M_PI / 6, 4 * M_PI / 3, 3 * M_PI / 2, 5 * M_PI / 3, 11 * M_PI / 6 };
			for (int i = 0; i < 12; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}

		//计算该点的特征值
		float LDI_temp = 1;
		int bin_Idx;
		if (section2 == 1) {
			bin_Idx = 96 + section3 - 1;
		}
		if (section2 == 2)
		{
			bin_Idx = 8 * (section - 1) + section3 - 1;
		}
		if (section2 == 3)
		{
			bin_Idx = 8 * 8 + 4 * (section - 1) + section3 - 1;
		}
		bin_cloud_LDI1[bin_Idx].push_back(LDI_temp);
	}
	sub_histogram1.resize(108);
	for (int i = 0; i < 108; i++)
	{
		if (bin_cloud_LDI1[i].size() == 0) sub_histogram1[i] = 0;//如果该方向没有贡献，则为0
		else
		{
			int sum = std::accumulate(bin_cloud_LDI1[i].begin(), bin_cloud_LDI1[i].end(), 0);//将这个方向的所有的贡献加起来
			sub_histogram1[i] = sum;
		}
	}
	//归一化
	float sum_sub_1 = std::accumulate(sub_histogram1.begin(), sub_histogram1.end(), 0.0f);
	for (float& val : sub_histogram1) {
		val /= sum_sub_1;
	}



	//View2，投影到yz平面----------------------------------------------------------------------------------------------------------
	std::vector<std::vector<float>> bin_cloud_LDI2;
	bin_cloud_LDI2.resize(108);
	for (int i = 0; i < number_of_points; i++)
	{
		//calculate bin's index
		//将投影坐标转换为极坐标，确定它在哪个圆圈中
		float rou = std::sqrt(cloud->points[i].y * cloud->points[i].y + cloud->points[i].z * cloud->points[i].z);
		double theta = std::atan2(cloud->points[i].z, cloud->points[i].y);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta = std::fmod(theta + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta < 0) {
			theta += 2 * Pi;
		}
		// 将角度映射到 8 个部分
		int section = static_cast<int>(theta / (Pi / 4)) + 1;
		if (section > 8) {
			section = 1;
		}
		int section2 = rou / (sup_radius / 5);
		section2 += 1;
		if (section2 == 3)
			section2 = 2;
		else if (section2 == 4 || section2 == 5)
			section2 = 3;


		//计算法向量的贡献方向
		int section3;
		double minDiff = std::numeric_limits<double>::max();//法向量与贡献方向的夹角

		//计算投影法向量的方向
		double theta2 = std::atan2(normals->points[i].normal_z, normals->points[i].normal_y);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta2 = std::fmod(theta2 + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta2 < 0) {
			theta2 += 2 * Pi;
		}

		if (section2 == 3) {
			// 计算与四个方向之间的差值
			double directions[4] = { 0, M_PI / 2, M_PI, 3 * M_PI / 2 };
			for (int i = 0; i < 4; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else if (section2 == 2)
		{
			// 计算与八个方向之间的差值
			double directions[8] = { 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI, 5 * M_PI / 4, 3 * M_PI / 2, 7 * M_PI / 4 };
			for (int i = 0; i < 8; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else {
			// 计算与十二个方向之间的差值
			double directions[12] = { 0, M_PI / 6, M_PI / 3, M_PI / 2, 2 * M_PI / 3, 5 * M_PI / 6, M_PI, 7 * M_PI / 6, 4 * M_PI / 3, 3 * M_PI / 2, 5 * M_PI / 3, 11 * M_PI / 6 };
			for (int i = 0; i < 12; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}

		//计算该点的特征值
		float LDI_temp = 1;
		int bin_Idx;
		if (section2 == 1) {
			bin_Idx = 96 + section3 - 1;
		}
		if (section2 == 2)
		{
			bin_Idx = 8 * (section - 1) + section3 - 1;
		}
		if (section2 == 3)
		{
			bin_Idx = 8 * 8 + 4 * (section - 1) + section3 - 1;
		}
		bin_cloud_LDI2[bin_Idx].push_back(LDI_temp);
	}
	sub_histogram2.resize(108);
	for (int i = 0; i < 108; i++)
	{
		if (bin_cloud_LDI2[i].size() == 0) sub_histogram2[i] = 0;//如果该方向没有贡献，则为0
		else
		{
			int sum = std::accumulate(bin_cloud_LDI2[i].begin(), bin_cloud_LDI2[i].end(), 0);//将这个方向的所有的贡献加起来
			sub_histogram2[i] = sum;
		}
	}
	//归一化
	float sum_sub_2 = std::accumulate(sub_histogram2.begin(), sub_histogram2.end(), 0.0f);
	for (float& val : sub_histogram2) {
		val /= sum_sub_2;
	}



	//View3,投影到xz平面----------------------------------------------------------------------------------------
	std::vector<std::vector<float>> bin_cloud_LDI3;
	bin_cloud_LDI3.resize(108);
	for (int i = 0; i < number_of_points; i++)
	{
		//calculate bin's index
		//将投影坐标转换为极坐标，确定它在哪个圆圈中
		float rou = std::sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].z * cloud->points[i].z);
		double theta = std::atan2(cloud->points[i].z, cloud->points[i].x);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta = std::fmod(theta + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta < 0) {
			theta += 2 * Pi;
		}
		// 将角度映射到 8 个部分
		int section = static_cast<int>(theta / (Pi / 4)) + 1;
		if (section > 8) {
			section = 1;
		}
		int section2 = rou / (sup_radius / 5);
		section2 += 1;
		if (section2 == 3)
			section2 = 2;
		else if (section2 == 4 || section2 == 5)
			section2 = 3;


		//计算法向量的贡献方向
		int section3;
		double minDiff = std::numeric_limits<double>::max();//法向量与贡献方向的夹角

		//计算投影法向量的方向
		double theta2 = std::atan2(normals->points[i].normal_z, normals->points[i].normal_x);//先转换为极坐标
		// 将角度限制在 [-π, π] 范围内
		theta2 = std::fmod(theta2 + Pi, 2 * Pi) - Pi;
		// 将角度转换为 [0, 2π] 范围内的正值
		if (theta2 < 0) {
			theta2 += 2 * Pi;
		}

		if (section2 == 3) {
			// 计算与四个方向之间的差值
			double directions[4] = { 0, M_PI / 2, M_PI, 3 * M_PI / 2 };
			for (int i = 0; i < 4; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else if (section2 == 2)
		{
			// 计算与八个方向之间的差值
			double directions[8] = { 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI, 5 * M_PI / 4, 3 * M_PI / 2, 7 * M_PI / 4 };
			for (int i = 0; i < 8; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}
		else {
			// 计算与十二个方向之间的差值
			double directions[12] = { 0, M_PI / 6, M_PI / 3, M_PI / 2, 2 * M_PI / 3, 5 * M_PI / 6, M_PI, 7 * M_PI / 6, 4 * M_PI / 3, 3 * M_PI / 2, 5 * M_PI / 3, 11 * M_PI / 6 };
			for (int i = 0; i < 12; i++) {
				double diff = std::abs(theta2 - directions[i]);
				if (diff < minDiff) {
					minDiff = diff;
					section3 = i + 1;
				}
			}
		}

		//计算该点的特征值
		float LDI_temp = 1;
		int bin_Idx;
		if (section2 == 1) {
			bin_Idx = 96 + section3 - 1;
		}
		if (section2 == 2)
		{
			bin_Idx = 8 * (section - 1) + section3 - 1;
		}
		if (section2 == 3)
		{
			bin_Idx = 8 * 8 + 4 * (section - 1) + section3 - 1;
		}
		bin_cloud_LDI3[bin_Idx].push_back(LDI_temp);
	}
	sub_histogram3.resize(108);
	for (int i = 0; i < 108; i++)
	{
		if (bin_cloud_LDI3[i].size() == 0) sub_histogram3[i] = 0;//如果该方向没有贡献，则为0
		else
		{
			int sum = std::accumulate(bin_cloud_LDI3[i].begin(), bin_cloud_LDI3[i].end(), 0);//将这个方向的所有的贡献加起来
			sub_histogram3[i] = sum;
		}
	}
	//归一化
	float sum_sub_3 = std::accumulate(sub_histogram3.begin(), sub_histogram3.end(), 0.0f);
	for (float& val : sub_histogram3) {
		val /= sum_sub_3;
	}




	//Concatenate the three sub-histograms, generating the final TOLDI feature
	//串接三个投影平面的直方图，生成108*3维的直方图
	std::copy(sub_histogram2.begin(), sub_histogram2.end(), std::back_inserter(sub_histogram1));
	std::copy(sub_histogram3.begin(), sub_histogram3.end(), std::back_inserter(sub_histogram1));
	histogram = sub_histogram1;
}

//生成关键点索引处的HPVND描述符
void HPVND_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int>indices, float sup_radius, int bin_num, std::vector<std::vector<float>>& Histograms)
{
	int i, j, m;
	//kdtree-search
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>pointIdx;
	std::vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;

	for (i = 0; i < indices.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);//local surface局部表面
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_z(new pcl::PointCloud<pcl::PointXYZ>);//local surface for computing the z-axis of LRF计算z轴的局部表面
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_trans(new pcl::PointCloud<pcl::PointXYZ>);//transformed local surface w.r.t. to LRF
		Vertex x_axis, y_axis, z_axis;
		query_point = cloud->points[indices[i]];
		//
		if (kdtree.radiusSearch(query_point, sup_radius / 3, pointIdx, pointDst) > 3)
		{
			for (j = 0; j < pointIdx.size(); j++)
			{
				sphere_neighbor_z->points.push_back(cloud->points[pointIdx[j]]);
			}
			HPVND_LRF_Z_axis(sphere_neighbor_z, z_axis);
			if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 10)//only if there are more than 10 points in the local surface
			{
				for (j = 0; j < pointIdx.size(); j++)
				{
					sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);
				}
				HPVND_LRF_X_axis(sphere_neighbor, z_axis, sup_radius, pointDst, x_axis);
				HPVND_LRF_Y_axis(x_axis, z_axis, y_axis);
				LRF pointLRF = { indices[i],x_axis,y_axis,z_axis };
				transformCloud(sphere_neighbor, pointLRF, sphere_neighbor_trans);//transform the local surface w.r.t. TOLDI-LRF
				std::vector<float> TriLDI_feature;
				local_HPVND(sphere_neighbor_trans, sup_radius, bin_num, TriLDI_feature);
				Histograms.push_back(TriLDI_feature);
			}
			else
			{
				std::vector<float> f_default(3 * 108, 0.0f);
				Histograms.push_back(f_default);
			}
		}
		else
		{
			std::vector<float> f_default(3 * 108, 0.0f);
			Histograms.push_back(f_default);
		}
	}
}