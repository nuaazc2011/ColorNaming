#pragma once
#include<opencv2\opencv.hpp>
#include<iostream>  
#include<opencv2\core\core.hpp>  
#include<opencv2\highgui\highgui.hpp>  
using namespace cv;
using namespace std;
typedef Vec<double, 11> Vec11d;

//===========颜色命名的第一种方法;参考Learning Color Names for Real-World Applications===========//
//Black, Blue, Brown, Grey, Green, Orange, Pink, Purple, Red, White, Yellow
class ColorNamePLSA
{
private:
	vector<vector<double>> RGBtable;									//查找表，大小为32768*11;

	Mat floorMat(const Mat & doubleMat);								//注意参数只为 CV_64FC1  
public:
	ColorNamePLSA();													//构造函数，查找表初始化
	vector<double> SampleColorNamePLSA(double R, double G, double B);   //获得RGB空间的一点隶属于11种颜色的概率值
	Mat getColorMapPLSA(const Mat &src);								//得到输入图像的每个像素点的颜色可视化
	vector<double> getColorNameFeature(const Mat &src);					//采用LOMO特征类似的方法，得到的颜色特征为165维
};



//===============颜色命名的第二种方法;参考Parametric fuzzy sets for automatic color naming=====//
//Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White
class ColorNameTSE
{
private:
	int numColors;														//颜色类别数
	int numChromatics;													//彩色类别数
	int numAchromatics;													//非彩色类别数
	double parameters[6][8][10];										//parameters of the model(chromatic colors)
	int thrL[7];														//parameters of the model(lightness levels)
	double paramsAchro[4][2];											//parameters of the model(achromatic colors)

	static double Sigmoid(double s, double t, double b);
public:
	ColorNameTSE();														//构造函数，参数初始化
	vector<double> SampleColorNameTSE(double L, double a, double b);    //获得Lab空间的一点隶属于11种颜色的概率值
	Mat getColorMapTSE(const Mat &src);									//得到输入图像的每个像素点的颜色可视化
	vector<double> getColorNameFeature(const Mat &src);					//采用LOMO特征类似的方法，得到的颜色特征为165维
};
