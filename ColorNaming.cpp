#include"ColorNaming.h"
#include<fstream>

ColorNamePLSA::ColorNamePLSA()
{
	long rows = 32768;
	int cols = 14;
	vector<vector<double>>tmptable(rows, vector<double>(cols - 3, 0));
	RGBtable = tmptable;
	
	double tmp;
	ifstream infile("./colorNamePLSA/w2c.txt");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			infile >> tmp;
			if (j < 3)
				continue;
			else
				RGBtable[i][j - 3] = tmp;
		}
	}
}

vector<double> ColorNamePLSA::SampleColorNamePLSA(double R, double G, double B)
{
	int fr = R / 8.0;
	int fg = G / 8.0;
	int fb = B / 8.0;
	int index = fr + fg * 32 + fb * 32 * 32;

	vector<double> CD=RGBtable[index];
	return CD;
}

Mat ColorNamePLSA::floorMat(const Mat & doubleMat)
{
	int rows = doubleMat.rows;
	int cols = doubleMat.cols;
	Mat flo(rows, cols, CV_32SC1);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			flo.at<int>(r, c) = floor(doubleMat.at<double>(r, c) / 8.0);
		}
	}
	return flo;
}

Mat ColorNamePLSA::getColorMapPLSA(const Mat &src)
{
	Mat image = src.clone();
	if (!image.data)
		return image;
	if (image.channels() != 3)
		return image;

	int rows = image.rows;
	int cols = image.cols;
	int areas = rows*cols;

	//分离通道  
	vector<Mat> bgr_planes;
	split(image, bgr_planes);

	//把各通道转为64F  
	Mat bplanes, gplanes, rplanes;
	bgr_planes[0].convertTo(bplanes, CV_64FC1);
	bgr_planes[1].convertTo(gplanes, CV_64FC1);
	bgr_planes[2].convertTo(rplanes, CV_64FC1);

	//floor(各通道/8.0)  
	Mat fbplanes, fgplanes, frplanes;
	fbplanes = floorMat(bplanes);
	fgplanes = floorMat(gplanes);
	frplanes = floorMat(rplanes);
	Mat index_im = frplanes + 32 * fgplanes + 32 * 32 * fbplanes;//index_im最大值可能为:31+31*32+32*32*31=32767  

	Mat index_im_col;
	index_im_col = index_im.reshape(1, areas);

	//读取w2cM  
	FileStorage fs("./colorNamePLSA/w2cM.xml", FileStorage::READ);
	Mat w2cM;
	fs["w2cM"] >> w2cM;
	//index_col存放的0-10之间的数值，代表11种颜色  
	Mat index_col(areas, 1, CV_32SC1);
	int tempIndex;
	for (int r = 0; r< areas; r++)
	{
		tempIndex = index_im_col.at<int>(r, 0);
		index_col.at<int>(r, 0) = w2cM.at<int>(tempIndex, 0);
	}
	//reshape  
	Mat out2 = index_col.reshape(1, rows);
	/* ----------------------11种颜色值----------------------------------*/
	Mat color_values(11, 1, CV_64FC3);
	//black-黑色 [0 0 0]  
	color_values.at<Vec3d>(0, 0) = Vec3d(0, 0, 0);

	//blue-蓝色 [1 0 0]  
	color_values.at<Vec3d>(1, 0) = Vec3d(1, 0, 0);

	//brown-棕色[0.25 0.4 0.5]  
	color_values.at<Vec3d>(2, 0) = Vec3d(0.25, 0.4, 0.5);

	//grey-灰色[0.5 0.5 0.5]  
	color_values.at<Vec3d>(3, 0) = Vec3d(0.5, 0.5, 0.5);

	//green-绿色[0 1 0]  
	color_values.at<Vec3d>(4, 0) = Vec3d(0, 1, 0);

	//orange-橘色[0 0.8 1]  
	color_values.at<Vec3d>(5, 0) = Vec3d(0, 0.8, 1);

	//pink-粉红色[1 0.5 1]  
	color_values.at<Vec3d>(6, 0) = Vec3d(1, 0.5, 1);

	//purple-紫色[1 0 1]  
	color_values.at<Vec3d>(7, 0) = Vec3d(1, 0, 1);

	//red-红色 [0 0 1]  
	color_values.at<Vec3d>(8, 0) = Vec3d(0, 0, 1);

	//white-白色 [1 1 1]  
	color_values.at<Vec3d>(9, 0) = Vec3d(1, 1, 1);

	//yellow-黄色[0 1 1]  
	color_values.at<Vec3d>(10, 0) = Vec3d(0, 1, 1);

	Mat out(rows, cols, CV_64FC3);
	for (int r = 0; r<rows; r++)
	{
		for (int c = 0; c<cols; c++)
		{
			int tindex = out2.at<int>(r, c);
			out.at<Vec3d>(r, c) = color_values.at<Vec3d>(tindex, 0) * 255;
		}
	}
	Mat colorMap;
	out.convertTo(colorMap, CV_8UC3);
	//关闭文件  
	fs.release();
	return colorMap;
}

vector<double> ColorNamePLSA::getColorNameFeature(const Mat &src)
{
	Mat img = src.clone();
	const int colorChannel = 9;					//9通道，即9种颜色
	int rows = 160;								//图像缩放大小
	int cols = 200;
	Size size = Size(cols, rows);
	resize(img, img, size);

	int blocksize = 20;							//每个patch的大小
	int strideX = 10;
	int strideY = 10;

	Mat proImg(rows, cols, CV_64FC(11));		//11通道，即11种颜色
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double B = img.at<Vec3b>(i, j)[0];
			double G = img.at<Vec3b>(i, j)[1];
			double R = img.at<Vec3b>(i, j)[2];
			vector<double> pro = SampleColorNamePLSA(R, G, B);
			double* arraypro = new double[pro.size()];
			for (int k = 0; k<pro.size(); k++)
				arraypro[k] = pro.at(k);
			Vec11d a(arraypro);

			proImg.at<Vec11d>(i, j) = a;
			delete[]arraypro;
		}
	}

	vector<double> feat;
	for (int i = 0; i <= rows - blocksize; i = i + strideY)
	{
		vector<Vec11d> everyLine;//采用LOMO类似的方法求取每一行的颜色特征，每一行的颜色特征维数为colorChannel
		for (int j = 0; j <= cols - blocksize; j = j + strideX)
		{
			Rect rect(j, i, blocksize, blocksize);
			Mat blockROI = proImg(rect);

			Vec11d tmp;
			for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
			{
				for (int i1 = 0; i1 < blocksize; i1++)
				{
					for (int j1 = 0; j1 < blocksize; j1++)
					{
						tmp[channel] += blockROI.at<Vec11d>(i1, j1)[channel];
					}
				}
			}
			everyLine.push_back(tmp);
		}

		Vec11d featLine;
		for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
		{
			featLine[channel] = -1;
			for (int k = 0; k < everyLine.size(); k++)
			{
				if (everyLine[k][channel]>featLine[channel])
					featLine[channel] = everyLine[k][channel];
			}
		}
		//featLine /= (blocksize*blocksize);

		for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
		{
			feat.push_back(featLine[channel]);
		}
	}

	double sum = 0;
	for (int i = 0; i < feat.size(); i++)
		sum += feat[i] * feat[i];
	for (int i = 0; i < feat.size(); i++)//归一化处理
		feat[i] = feat[i] / sqrt(sum);

	return feat;
}





ColorNameTSE::ColorNameTSE()
{
	double tmpparameters[6][8][10] =
	{
		{
			{ 0.4242, 0.2494, -0.0391, -0.9870, 0.9010, 1.7250, 9.8441, 5.8894, 7.4743, 0.0405 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.4242, 0.2494, 0.5838, 0.2540, 1.7250, 0.8440, 9.8441, 5.8894, 7.4743, 0.0405 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.4242, 0.2494, 1.8248, 2.3490, 0.8440, 1.9522, 9.8441, 5.8894, 7.4743, 0.0405 },
			{ 0.4242, 0.2494, 3.9198, -2.5683, 1.9522, 1.0143, 9.8441, 5.8894, 7.4743, 0.0405 },
			{ 0.4242, 0.2494, -0.9975, -1.6099, 1.0143, 0.9010, 9.8441, 5.8894, 7.4743, 0.0405 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		},
		{
			{ 0.2296, 0.6632, 0.0385, -0.8520, 0.5244, 5.0000, 6.0326, 6.4571, 7.8725, 0.3070 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.2296, 0.6632, 0.7188, 0.1199, 5.0000, 0.6924, 6.0326, 6.4571, 7.8725, 0.3070 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.2296, 0.6632, 1.6907, 2.1025, 0.6924, 0.9581, 6.0326, 6.4571, 7.8725, 0.3070 },
			{ 0.2296, 0.6632, 3.6733, -2.5915, 0.9581, 0.9166, 6.0326, 6.4571, 7.8725, 0.3070 },
			{ 0.2296, 0.6632, -1.0207, -1.8452, 0.9166, 1.1003, 6.0326, 6.4571, 7.8725, 0.3070 },
			{ 0.2296, 0.6632, -0.2744, -1.5323, 1.1003, 0.5244, 6.0326, 6.4571, 7.8725, 0.3070 }
		},
		{
			{ -0.1173, 0.5180, 0.2369, -0.7949, 0.9994, 0.5653, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ -0.1173, 0.5180, 0.7759, -0.5020, 0.5653, 0.5172, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ -0.1173, 0.5180, 1.0688, 0.1161, 0.5172, 0.8448, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.1173, 0.5180, 1.6869, 1.9090, 0.8448, 0.5998, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ -0.1173, 0.5180, 3.4798, -2.5873, 0.5998, 0.8008, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ -0.1173, 0.5180, -1.0165, -1.9658, 0.8008, 0.6237, 6.8122, 5.3811, 6.9824, 0.3418 },
			{ -0.1173, 0.5180, -0.3950, -1.3339, 0.6237, 0.9994, 6.8122, 5.3811, 6.9824, 0.3418 },

		},
		{
			{ -0.4432, 1.0759, 0.4629, -0.9950, 0.9391, 0.7548, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ -0.4432, 1.0759, 0.5758, -0.1725, 0.7548, 0.4784, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ -0.4432, 1.0759, 1.3983, 0.0990, 0.4784, 0.7287, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ -0.4432, 1.0759, 1.6698, 1.8905, 0.7287, 0.6438, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ -0.4432, 1.0759, 3.4613, -2.5966, 0.6438, 0.7552, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ -0.4432, 1.0759, -1.0258, -2.1629, 0.7552, 5.0000, 7.3186, 6.0637, 7.5020, 0.4284 },
			{ -0.4432, 1.0759, -0.5921, -1.1079, 5.0000, 0.9391, 7.3186, 6.0637, 7.5020, 0.4284 }
		},
		{
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ -0.5655, 1.1633, 0.4494, -0.2766, 1.9951, 0.8399, 100.0000, 5.3682, 6.9005, 0.4320 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 0.5655, 1.1633, 1.2942, 0.2142, 0.8399, 0.8560, 100.0000, 5.3682, 6.9005, 0.4320 },
			{ -0.5655, 1.1633, 1.7850, 1.7203, 0.8560, 0.7366, 100.0000, 5.3682, 6.9005, 0.4320 },
			{ -0.5655, 1.1633, 3.2911, -2.6325, 0.7366, 0.4748, 100.0000, 5.3682, 6.9005, 0.4320 },
			{ -0.5655, 1.1633, -1.0617, -2.1389, 0.4748, 1.7356, 100.0000, 5.3682, 6.9005, 0.4320 },
			{ -0.5655, 1.1633, -0.5681, -1.1214, 1.7356, 1.9951, 100.0000, 5.3682, 6.9005, 0.4320 }
		},
		{
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ 1.2564, 1.8144, 0.4493, -0.3064, 1.0280, 0.7910, 100.0000, 6.0399, 7.3932, -0.0208 },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			{ -1.2564, 1.8144, 1.2644, 0.2835, 0.7910, 0.9574, 100.0000, 6.0399, 7.3932, -0.0208 },
			{ -1.2564, 1.8144, 1.8543, 1.7461, 0.9574, 0.9000, 100.0000, 6.0399, 7.3932, -0.0208 },
			{ -1.2564, 1.8144, 3.3169, -2.6081, 0.9000, 0.5990, 100.0000, 6.0399, 7.3932, -0.0208 },
			{ -1.2564, 1.8144, -1.0373, -2.1357, 0.5990, 1.9313, 100.0000, 6.0399, 7.3932, -0.0208 },
			{ -1.2564, 1.8144, -0.5649, -1.1215, 1.9313, 1.0280, 100.0000, 6.0399, 7.3932, -0.0208 }
		}
	};
	int tmpthrL[7] = { 0, 31, 42, 51, 66, 76, 150 };
	double tmpparamsAchro[4][2] = { { 28.2825, -0.7142 }, { 28.2825, 0.7142 }, { 79.6493, -0.3067 }, { 79.6493, 0.3067 } };

	numColors = 11;                             // Number of colors
	numAchromatics = 3;                         // Number of achromatic colors
	numChromatics = numColors - numAchromatics; // Number of chromatic colors
	
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 10; k++)
				parameters[i][j][k] = tmpparameters[i][j][k];
		}
	}

	for (int i = 0; i < 7; i++)
		thrL[i] = tmpthrL[i];

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			paramsAchro[i][j] = tmpparamsAchro[i][j];
		}
	}
}


double ColorNameTSE::Sigmoid(double s, double t, double b)
{
	double y = 1.0 / (1 + exp(-b*(s - t)));
	return y;
}

//SampleColorNameTSE: Given a sLab sample, returns the color name assigned to the 11 memberships to the 11 basic colors(in 'CD')
//The membership values in the CD are ordered : Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White
//s - Sample in sLab format[L a b]
vector<double> ColorNameTSE::SampleColorNameTSE(double L, double a, double b)
{
	int numLevels = sizeof(thrL) / sizeof(int) - 1;
	vector<double> CD(numColors, 0);

	int m = 0;
	for (int i = 0; i < numLevels; i++)
	{
		if (thrL[i] <= L&&L < thrL[i + 1])
			m = i;
	}

	for (int k = 0; k < numChromatics; k++)
	{
		double tx = parameters[m][k][0];
		double ty = parameters[m][k][1];
		double alfa_x = parameters[m][k][2];
		double alfa_y = parameters[m][k][3];
		double beta_x = parameters[m][k][4];
		double beta_y = parameters[m][k][5];
		double beta_e = parameters[m][k][6];
		double ex = parameters[m][k][7];
		double ey = parameters[m][k][8];
		double angle_e = parameters[m][k][9];
		if (beta_e != 0.0)
		{
			double a1 = a - tx;
			double b1 = b - ty;
			double sR1 = a1*cos(alfa_y) + b1*sin(alfa_y);
			double sR2 = -a1*sin(alfa_x) + b1*cos(alfa_x);

			double sC1 = a1*cos(angle_e) + b1*sin(angle_e);
			double sC2 = -a1*sin(angle_e) + b1*cos(angle_e);

			if (ex == 0.0)
				ex = 1;
			if (ey == 0.0)
				ey = 1;

			double y1 = 1.0 / (1 + exp(-sR1*beta_y));
			double y2 = 1.0 / (1 + exp(-sR2*beta_x));	
			double y3 = 1.0/(1 + exp(-beta_e*((sC1 / ex)*(sC1 / ex)+(sC2 / ey)*(sC2 / ey) - 1)));
			double y = y1*y2*y3;

			CD[k] = y;
		}
		else
		{
			CD[k] = 0;
		}
	}
	double colorSum = 0;
	for (int i = 0; i < numColors; i++)
		colorSum += CD[i];
	double valueAchro = (1 - colorSum>0.0) ? (1 - colorSum) : 0.0;
	CD[numChromatics] = valueAchro*Sigmoid(L, paramsAchro[0][0], paramsAchro[0][1]);
	CD[numChromatics + 1] = valueAchro*Sigmoid(L, paramsAchro[1][0], paramsAchro[1][1])*Sigmoid(L, paramsAchro[2][0], paramsAchro[2][1]);
	CD[numChromatics + 2] = valueAchro*Sigmoid(L, paramsAchro[3][0], paramsAchro[3][1]);
	
	return CD;
}

Mat ColorNameTSE::getColorMapTSE(const Mat &src)
{
	Mat image3f;
	src.convertTo(image3f, CV_32F, 1.0 / 255);
	Mat Lab;
	cvtColor(image3f, Lab, CV_BGR2Lab);

	int rows = src.rows;
	int cols = src.cols;

	Mat mask(src.rows, src.cols, CV_8UC1);
	mask.setTo(0);
	Mat colormask(src.rows, src.cols, CV_8UC3);
	colormask.setTo(0);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double L = Lab.at<Vec3f>(i, j)[0];
			double a = Lab.at<Vec3f>(i, j)[1];
			double b = Lab.at<Vec3f>(i, j)[2];
			vector<double> CD = SampleColorNameTSE(L,a,b);

			vector<double>::iterator biggest = max_element(std::begin(CD), std::end(CD));
			int code=distance(std::begin(CD), biggest);		
			mask.at<uchar>(i, j) = code;
		}
	}
	//Red, Orange, Brown, Yellow, Green, Blue, Purple, Pink, Black, Grey, White
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int label = mask.at<uchar>(i, j);
			switch (label)
			{
			case 0:
				colormask.at<Vec3b>(i, j) = Vec3b(0, 0, 255);//Red
				break;
			case 1:
				colormask.at<Vec3b>(i, j) = Vec3b(0, 255*0.8, 255);//Orange
				break;
			case 2:
				colormask.at<Vec3b>(i, j) = Vec3b(255 * 0.25, 255 * 0.4, 255*0.5);//Brown
				break;
			case 3:
				colormask.at<Vec3b>(i, j) = Vec3b(0, 255, 255);//yellow
				break;
			case 4:
				colormask.at<Vec3b>(i, j) = Vec3b(0, 255, 0);//Green
				break;
			case 5:
				colormask.at<Vec3b>(i, j) = Vec3b(255, 0, 0);//Blue
				break;
			case 6:
				colormask.at<Vec3b>(i, j) = Vec3b(255, 0, 255);//Purple
				break;
			case 7:
				colormask.at<Vec3b>(i, j) = Vec3b(255, 0.5*255, 255);// Pink
				break;
			
			case 8:
				colormask.at<Vec3b>(i, j) = Vec3b(0, 0, 0);// black
				break;
			case 9:
				colormask.at<Vec3b>(i, j) = Vec3b(125, 125, 125);// gray
				break;
			case 10:
				colormask.at<Vec3b>(i, j) = Vec3b(255, 255, 255);// white
				break;
			default:
				break;
			}
		}
	}
	return colormask;
}

vector<double> ColorNameTSE::getColorNameFeature(const Mat &src)
{
	Mat img = src.clone();
	const int colorChannel = 9;					//9通道，即9种颜色
	int rows = 160;								//图像缩放大小
	int cols = 200;
	Size size = Size(cols, rows);
	resize(img, img, size);

	int blocksize = 20;							//每个patch的大小
	int strideX = 10;
	int strideY = 10;

	Mat image3f;
	img.convertTo(image3f, CV_32F, 1.0 / 255);
	Mat Lab;
	cvtColor(image3f, Lab, CV_BGR2Lab);

	Mat proImg(rows, cols, CV_64FC(11));		//11通道，即11种颜色
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double L = Lab.at<Vec3f>(i, j)[0];
			double a = Lab.at<Vec3f>(i, j)[1];
			double b = Lab.at<Vec3f>(i, j)[2];
			vector<double> pro = SampleColorNameTSE(L, a, b);
			double* arraypro = new double[pro.size()];
			for (int k = 0; k<pro.size(); k++)
				arraypro[k] = pro.at(k);
			Vec11d val(arraypro);

			proImg.at<Vec11d>(i, j) = val;
			delete[]arraypro;
		}
	}

	vector<double> feat;
	for (int i = 0; i <= rows - blocksize; i = i + strideY)
	{
		vector<Vec11d> everyLine;//采用LOMO类似的方法求取每一行的颜色特征，每一行的颜色特征维数为colorChannel
		for (int j = 0; j <= cols - blocksize; j = j + strideX)
		{
			Rect rect(j, i, blocksize, blocksize);
			Mat blockROI = proImg(rect);

			Vec11d tmp;
			for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
			{
				for (int i1 = 0; i1 < blocksize; i1++)
				{
					for (int j1 = 0; j1 < blocksize; j1++)
					{
						tmp[channel] += blockROI.at<Vec11d>(i1, j1)[channel];
					}
				}
			}
			everyLine.push_back(tmp);
		}

		Vec11d featLine;
		for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
		{
			featLine[channel] = -1;
			for (int k = 0; k < everyLine.size(); k++)
			{
				if (everyLine[k][channel]>featLine[channel])
					featLine[channel] = everyLine[k][channel];
			}
		}
		//featLine /= (blocksize*blocksize);

		for (int channel = 0; channel < colorChannel; channel++)//11通道，即11种颜色
		{
			feat.push_back(featLine[channel]);
		}
	}

	double sum = 0;
	for (int i = 0; i < feat.size(); i++)
		sum += feat[i] * feat[i];
	for (int i = 0; i < feat.size(); i++)//归一化处理
		feat[i] = feat[i] / sqrt(sum);
	return feat;
}