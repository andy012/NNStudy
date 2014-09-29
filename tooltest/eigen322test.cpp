
#include "stdafx.h"

#include <iostream>  
#include <Eigen/Dense> 
using namespace std;
//using Eigen::MatrixXd;  
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;  
//定义非并行
#define  EIGEN_DONT_PARALLELIZE

/*
关于并行运算：
http://eigen.tuxfamily.org/dox/TopicMultiThreading.html
关于入门知识：
http://www.cnblogs.com/cyxcw1/archive/2013/04/28/3051288.html
*/

int _tmain(int argc, _TCHAR* argv[])
{
	  cout<<"*******************1D-object****************"<<endl;  
  
    Vector4d v1;  
    v1<< 1,2,3,4;  
    cout<<"v1=\n"<<v1<<endl;  
	VectorXd v2(3);  
    v2<<1,2,3;  
    cout<<"v2=\n"<<v2<<endl;  
  
    Array4i v3;  
    v3<<1,2,3,4;  
    cout<<"v3=\n"<<v3<<endl;  
  
    ArrayXf v4(3);  
    v4<<1,2,3;  
    cout<<"v4=\n"<<v4<<endl;  
    cout<<"*******************2D-object****************"<<endl;  
    //2D objects:  
    MatrixXd m(2,2);  
  
    //method 1  
    m(0,0) = 3;  
    m(1,0) = 2.5;  
    m(0,1) = -1;  
    m(1,1) = m(1,0) + m(0,1);  
  cout <<"m=\n"<< m << endl;  
    //method 2  
    m<<3,-1,  
        2.5,-1.5;  
    cout <<"m=\n"<< m << endl;  
	while(true){}
	return 0;
}

