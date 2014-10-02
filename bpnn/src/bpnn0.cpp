/*
Author: Andy Doo
time:2014/10/2
detail:
c++版BP神经网络，使用的案例是：语音特征信号分类（matlab 神经网络43个案例分析第一章）
数据在data文件夹下
*/
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <float.h>
using namespace std;
#define LENGTH_ARRAY 2000
#define TRAIN_LENGTH_ARRAY 1500
#define TEST_LENGTH_ARRAY 500
#define INPUT_LAY_NUM 24
#define OUTPUT_LAY_NUM 4
#define MID_LAY_NUM 25
#define TRAIN_ITERATOR_TIMES 10
typedef float T;
typedef float D;



struct random_sort_pair{
	int index;
	int rand_value;
};
struct MinMax{
	T* max;
	T* min;
};
struct BPNN{

	int length_array;
	int train_length_array;
	int test_length_array;
	int input_lay_num;
	int mid_lay_num;
	int output_lay_num;
	int train_iterator_times;
	//样本输入值，输出值
	T input[LENGTH_ARRAY][INPUT_LAY_NUM];
	T output[LENGTH_ARRAY][OUTPUT_LAY_NUM];
	//训练样本
	T input_train[TRAIN_LENGTH_ARRAY][INPUT_LAY_NUM];
	T output_train[TRAIN_LENGTH_ARRAY][OUTPUT_LAY_NUM];
	//测试样本
	T input_test[TEST_LENGTH_ARRAY][INPUT_LAY_NUM];
	T output_test[TEST_LENGTH_ARRAY][OUTPUT_LAY_NUM];
	//权值和阈值
	D w1[MID_LAY_NUM][INPUT_LAY_NUM];
	D b1[MID_LAY_NUM];
	D w2[OUTPUT_LAY_NUM][MID_LAY_NUM];
	D b2[OUTPUT_LAY_NUM];
	D n;
}bpnn;

T active_function(T x);
void get_rand_pair_sorts_truct();
void midlay_output(T *train_data,T* midlay_output_data);
void outputlay_output(T*midlay_output_data,T* outputlay_output_data);
void e_output(T*outputlay_output_data,T*y_output_data,T*e);
void adapter_weight_range(T *train_data,T *midlay_output_data,T *e,T *temp_midlay_output_adpater,T*,T*);
struct random_sort_pair randArray[LENGTH_ARRAY];

//随机选择训练数据和测试数据
void init_train_test_rand(){
	get_rand_pair_sorts_truct();
	int i=0,j=0;
	for(i=0;i<bpnn.train_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input_train[i][j]=bpnn.input[randArray[i].index][j];
		}
		for(int k=0;k<bpnn.output_lay_num;++k){
			bpnn.output_train[i][k]=bpnn.output[randArray[i].index][k];
		}
	}
	int temp=bpnn.train_length_array;
	for(i=0;i<bpnn.test_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input_test[i][j]=bpnn.input[randArray[i+temp].index][j];
		}
		for(int k=0;k<bpnn.output_lay_num;++k){
			bpnn.output_test[i][k]=bpnn.output[randArray[i+temp].index][k];
		}
	}
	
	for(i=0;i<bpnn.train_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input[i][j]=bpnn.input_train[i][j];
		}
		for(int k=0;k<bpnn.output_lay_num;++k){
			bpnn.output[i][k]=bpnn.output_train[i][k];
		}
	}
	
	for(i=0;i<bpnn.test_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input[i+temp][j]=bpnn.input_test[i][j];
		}
		for(int k=0;k<bpnn.output_lay_num;++k){
			bpnn.output[i+temp][k]=bpnn.output_test[i][k];
		}
	}
	
}

int compare(const void* a, const void* b)
{
        struct random_sort_pair *a1 = (struct random_sort_pair *) a;
        struct random_sort_pair *a2 = (struct random_sort_pair *) b;
		return a1->rand_value - a2->rand_value;
}

//产生随机序列
void get_rand_pair_sorts_truct(){
	int i=0;
	//
	for(i=0;i<bpnn.length_array;++i){
	    randArray[i].index=i;
		randArray[i].rand_value=rand();
	}
//	for(i=0;i<length;++i){
//	    cout<<randArray[i].index<<"    "<<randArray[i].rand_value<<endl;
//	}
	qsort(randArray, bpnn.length_array, sizeof(random_sort_pair), compare);
//	for(i=0;i<length;++i){
//	    cout<<randArray[i].index<<"    "<<randArray[i].rand_value<<endl;
//	}

	
	FILE *file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\randindex.txt","w");
	for(i=0;i<bpnn.length_array;++i){
		fprintf(file,"%d ",randArray[i].index);
	}
	fclose(file);
	
}

int get_flag(){
    if(rand()%2==0) return 1;
	else return -1;
}

//初始化权值和阈值
void init_weight_b_rand(){
	int i=0,j=0;
	for(i=0;i<bpnn.mid_lay_num;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
		    bpnn.w1[i][j]=get_flag()*((float)rand())/ RAND_MAX;
		}
	}
	for(j=0;j<bpnn.mid_lay_num;++j){
		bpnn.b1[j]=get_flag()*((float)rand())/ RAND_MAX;
	}
	for(i=0;i<bpnn.output_lay_num;++i){
		for(j=0;j<bpnn.mid_lay_num;++j){
		    bpnn.w2[i][j]=get_flag()*((float)rand())/ RAND_MAX;
		}
	}
	for(j=0;j<bpnn.output_lay_num;++j){
		bpnn.b2[j]=get_flag()*((float)rand())/ RAND_MAX;
	}
	FILE *file=fopen("weight_b.txt","w");
	cout<<"初始化权值和阈值："<<endl;
	cout<<"w1"<<endl;
	fprintf(file,"%s\n","w1");
	for( i=0;i<bpnn.mid_lay_num;++i){
		for( j=0;j<bpnn.input_lay_num;++j){
			 fprintf(file,"%f ",(bpnn.w1[i][j]));
			cout<<bpnn.w1[i][j]<<"  ";
		}
		fprintf(file,"\n");
	}
	cout<<"b1"<<endl;
	fprintf(file,"%s\n","b1");
	for( i=0;i<bpnn.mid_lay_num;++i){
		fprintf(file,"%f\n",(bpnn.b1[i]));
	//cout<<bpnn.b1[i]<<endl;
	}
	cout<<endl;
	cout<<"w2"<<endl;
	fprintf(file,"%s\n","w2");
	for( i=0;i<bpnn.output_lay_num;++i){
		for( j=0;j<bpnn.mid_lay_num;++j){
			 fprintf(file,"%f ",(bpnn.w2[i][j]));
			cout<<bpnn.w2[i][j]<<"  ";
		}
		fprintf(file,"\n");
	}
	cout<<"b2"<<endl;
	fprintf(file,"\n%s\n","b2");
	for( i=0;i<bpnn.output_lay_num;++i){
		fprintf(file,"%f\n",(bpnn.b2[i]));
		cout<<bpnn.b2[i]<<endl;
		
	}
	fclose(file);
	/**/
}

//训练数据归一化
struct MinMax mapminmax(){
	struct MinMax mm;
	mm.max=new T[bpnn.input_lay_num];
	mm.min=new T[bpnn.input_lay_num];
	int i,j=0;
	for(i=0;i<bpnn.input_lay_num;++i){
		mm.max[i]= bpnn.input[0][i];
		mm.min[i]= bpnn.input[0][i];
		for(j=0;j<bpnn.train_length_array;++j){
			if(bpnn.input[j][i]<mm.min[i]){
				mm.min[i]=bpnn.input[j][i];
			}
			if(bpnn.input[j][i]>mm.max[i]){
				mm.max[i]=bpnn.input[j][i];
			}
		}
	}

	for(i=0;i<bpnn.input_lay_num;++i){
	    cout<<mm.max[i]<<"    "<<mm.min[i]<<endl;
	}

	for(i=0;i<bpnn.train_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input_train[i][j]=2*(bpnn.input[i][j]-mm.min[j])/(mm.max[j]-mm.min[j])-1;
		}
	}
	for(i=0;i<bpnn.train_length_array;++i){
		
		for(j=0;j<bpnn.output_lay_num;++j){
			bpnn.output_train[i][j]=bpnn.output[i][j];
		}
	}
	for(i=0;i<bpnn.test_length_array;++i){
		for(j=0;j<bpnn.input_lay_num;++j){
			bpnn.input_test[i][j]=2*(bpnn.input[i+bpnn.train_length_array][j]-mm.min[j])/(mm.max[j]-mm.min[j])-1;
		}
	}
	for(i=0;i<bpnn.test_length_array;++i){
		
		for(j=0;j<bpnn.output_lay_num;++j){
			bpnn.output_test[i][j]=bpnn.output[i+bpnn.train_length_array][j];
		}
	}
	/*
	FILE *file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\input__.txt","w");
	for(i=0;i<bpnn.train_length_array;++i){
		
		for(j=0;j<bpnn.input_lay_num;++j){
			fprintf(file,"%f ",bpnn.input[i][j]);
			//bpnn.input_train[i][j]=(bpnn.input[i][j]-mm.min[j])/(mm.max[j]-mm.min[j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
	file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\input_train.txt","w");
	for(i=0;i<bpnn.train_length_array;++i){
		
		for(j=0;j<bpnn.input_lay_num;++j){
			fprintf(file,"%f ",bpnn.input_train[i][j]);
			//bpnn.input_train[i][j]=(bpnn.input[i][j]-mm.min[j])/(mm.max[j]-mm.min[j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
	file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\output_train.txt","w");
	for(i=0;i<bpnn.train_length_array;++i){
		
		for(j=0;j<bpnn.output_lay_num;++j){
			fprintf(file,"%f ",bpnn.output_train[i][j]);
			//bpnn.input_train[i][j]=(bpnn.input[i][j]-mm.min[j])/(mm.max[j]-mm.min[j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);*/
	return mm;



}
//激励函数
T active_function(T x){
	return 1.0/(1.0+exp(-x));
}

//训练神经网络
void bpnn_train(){
	//训练次数
	int train_times=bpnn.train_iterator_times;
	//隐含层输出
	T *midlay_output_data =new T[bpnn.mid_lay_num];
	//输出层输出
	T *outputlay_output_data=new T[bpnn.output_lay_num];
	//预测误差
	T *e=new T[bpnn.output_lay_num];
	//
	T *temp_midlay_output_adpater=new T[bpnn.mid_lay_num];
	for(int train_i=0;train_i<train_times;++train_i){
		for(int train_data_i=0;train_data_i<bpnn.train_length_array;++train_data_i){
			midlay_output(bpnn.input_train[train_data_i],midlay_output_data);
			outputlay_output(midlay_output_data,outputlay_output_data);
			e_output(outputlay_output_data,bpnn.output_train[train_data_i],e);
			adapter_weight_range(bpnn.input_train[train_data_i],midlay_output_data,e,temp_midlay_output_adpater,outputlay_output_data,bpnn.output_train[train_data_i]);
		}
	}

	

	cout<<"训练后权值和阈值"<<endl;
	cout<<"w1"<<endl;
	int i,j,k;
	for( i=0;i<bpnn.mid_lay_num;++i){
		for( j=0;j<bpnn.input_lay_num;++j){
		    cout<<bpnn.w1[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"b1"<<endl;
	for( i=0;i<bpnn.mid_lay_num;++i){	
	cout<<bpnn.b1[i]<<" ";
	}
	cout<<endl;
	cout<<"w2"<<endl;
	for( i=0;i<bpnn.output_lay_num;++i){
		for( j=0;j<bpnn.mid_lay_num;++j){
		    cout<<bpnn.w2[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"b2"<<endl;
	for( i=0;i<bpnn.output_lay_num;++i){
		    cout<<bpnn.b2[i]<<" ";
	}
	cout<<endl;
	
	delete midlay_output_data;
	delete outputlay_output_data;
	delete e;
	delete temp_midlay_output_adpater;
	
}
//调整权值和阈值
void adapter_weight_range(T *train_data,T *midlay_output_data,T *e,T *temp_midlay_output_adpater,T *outputlay_output_data,T *output_y){
    int i,j,k;
	/*
	//输入值(x1,x2)
	cout<<"输入值(x1,x2)"<<endl;
	for(i=0;i<bpnn.input_lay_num;++i){
		cout<<"x"<<i<<"="<<train_data[i]<<"  ";
	}
	cout<<endl;
	//H隐含层输出：
	cout<<"隐含层输出:"<<endl;
	for(i=0;i<bpnn.mid_lay_num;++i){
		cout<<midlay_output_data[i]<<"    ";
	}
	cout<<endl;
	//输出层输出：
	cout<<"输出层输出："<<endl;
	for(i=0;i<bpnn.output_lay_num;++i){
		cout<<outputlay_output_data[i]<<"    ";
	}
	cout<<endl;
	//标准输出：
	cout<<"标准输出："<<endl;
	for(i=0;i<bpnn.output_lay_num;++i){
		cout<<output_y[i]<<"    ";
	}
	cout<<endl;
	//误差：
	cout<<"误差："<<endl;
	for(i=0;i<bpnn.output_lay_num;++i){
		cout<<e[i]<<"    ";
	}
	cout<<endl;
	//H(1-H)
	cout<<"H(1-H):"<<endl;
	for(i=0;i<bpnn.mid_lay_num;++i){
		temp_midlay_output_adpater[i]=bpnn.n*midlay_output_data[i]*(1-midlay_output_data[i]);
		cout<<temp_midlay_output_adpater[i]<<endl;
	}
	cout<<endl;
	*/
	//更新w1
	for(i=0;i<bpnn.mid_lay_num;++i){
		temp_midlay_output_adpater[i]=bpnn.n*midlay_output_data[i]*(1-midlay_output_data[i]);
		//cout<<temp_midlay_output_adpater[i]<<endl;
	}
	for(i=0;i<bpnn.input_lay_num;++i){
		for(j=0;j<bpnn.mid_lay_num;++j){
			D sum=0;
			for(int k=0;k<bpnn.output_lay_num;k++){
				sum+=bpnn.w2[k][j]*e[k];
			}
			bpnn.w1[j][i]=bpnn.w1[j][i]+train_data[i]*temp_midlay_output_adpater[j]*sum;
		}
	}
	//更新b1
	for(j=0;j<bpnn.mid_lay_num;++j){
		D sum=0;
		for( k=0;k<bpnn.output_lay_num;k++){
			sum+=bpnn.w2[k][j]*e[k];
		}
	    bpnn.b1[j]=bpnn.b1[j]+temp_midlay_output_adpater[j]*sum;
	}

	//更新w2
	for(j=0;j<bpnn.mid_lay_num;++j){
		for(k=0;k<bpnn.output_lay_num;++k){
		    bpnn.w2[k][j]= bpnn.w2[k][j]+bpnn.n*midlay_output_data[j]*e[k];
		}
	}
	//更新b2
	for(k=0;k<bpnn.output_lay_num;++k){
	    bpnn.b2[k]=bpnn.b2[k]+bpnn.n*e[k];
	}
	/**/
}

//计算隐含层的输出
void midlay_output(T *train_data,T* midlay_output_data){
	T sum;
	for(int i=0;i<bpnn.mid_lay_num;++i){
	    sum=0;
		for(int j=0;j<bpnn.input_lay_num;++j){
			sum+=train_data[j]*bpnn.w1[i][j];
		}
		midlay_output_data[i]=active_function(sum+bpnn.b1[i]);
	}
}

//计算输出层输出
void outputlay_output(T*midlay_output_data,T* outputlay_output_data){
	T sum=0;
	for(int i=0;i<bpnn.output_lay_num;++i){
		sum=0;
		for(int j=0;j<bpnn.mid_lay_num;++j){
		    sum+=midlay_output_data[j]*bpnn.w2[i][j];
		}
		outputlay_output_data[i]=sum+bpnn.b2[i];
	}
}
//计算输出误差
void e_output(T*outputlay_output_data,T*y_output_data,T*e){
	for(int i=0;i<bpnn.output_lay_num;++i){
	    e[i]=y_output_data[i]-outputlay_output_data[i];
	}
}
void read_data(){
	int i=0;
	FILE *file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\input.txt","r");
	
	for(i=0;i<bpnn.length_array;++i){
		for(int k=0;k<bpnn.input_lay_num;++k){
		    fscanf(file,"%f",&(bpnn.input[i][k]));
		}
	}

	//while(fscanf(file,"%f %f",&(bpnn.input[i][0]),&(bpnn.input[i][1]))!=EOF){
		//cout<<bpnn.input[i][0]<<"   "<<bpnn.input[i][1]<<endl;
	//	i++;
	//}
	fclose(file);
	file=fopen("C:\\Users\\Administrator\\Documents\\MATLAB\\output.txt","r");
	i=0;
	for(i=0;i<bpnn.length_array;++i){
		for(int k=0;k<bpnn.output_lay_num;++k){
		    fscanf(file,"%f",&(bpnn.output[i][k]));
		}
	}
	//while(fscanf(file,"%f",&(bpnn.output[i][0]))!=EOF){
	//	i++;
	//}
	fclose(file);
	/*
	for(i=0;i<LENGTH_ARRAY;++i){
		for(int j=0;j<INPUT_LAY_NUM;++j){
			cout<<bpnn.input[i][j]<<"  ";
		}
		cout<<"--------"<<endl;
		for(int k=0;k<bpnn.output_lay_num;++k){
		cout<<bpnn.output[i][k]<<"    ";
		}
		cout<<"####"<<endl;
	}
	int a ;
	cin>>a;
	*/
}
void init_bpnn_base(){
	bpnn.length_array=LENGTH_ARRAY;
	bpnn.train_length_array=TRAIN_LENGTH_ARRAY;
	bpnn.test_length_array=TEST_LENGTH_ARRAY;
	bpnn.input_lay_num=INPUT_LAY_NUM;
	bpnn.output_lay_num=OUTPUT_LAY_NUM;
	bpnn.mid_lay_num=MID_LAY_NUM;
	bpnn.train_iterator_times=TRAIN_ITERATOR_TIMES;
	bpnn.n=0.1;
	read_data();
	init_train_test_rand();
	struct MinMax mm=mapminmax();
	init_weight_b_rand();

}
void init(){
	init_bpnn_base();	
	
}

void print_output(T* res){
    for(int i=0;i<bpnn.output_lay_num;++i){
		cout<<res[i]<<" ";
	}
	cout<<endl;
}

T* compute_value(T* train_data){
	//隐含层输出
	T *midlay_output_data =new T[bpnn.mid_lay_num];
	//输出层输出
	T *outputlay_output_data=new T[bpnn.output_lay_num];
	midlay_output(train_data,midlay_output_data);
	outputlay_output(midlay_output_data,outputlay_output_data);
	//预测结果：
	delete midlay_output_data;
	return outputlay_output_data;

}

int find_index_of_max_number(T* myarray,int length){
	int index=0;
	T temp=myarray[0];
	for(int i=1;i<length;++i){
		if(temp<myarray[i]){
		    temp=myarray[i];
			index=i;
		}
	}
	return index;
}

void test_voice_bpnn(){
	int *prediction_ans=new int[bpnn.output_lay_num];
	int *correct_ans=new int[bpnn.output_lay_num];
	int i;
	for(i=0;i<bpnn.output_lay_num;++i){
		prediction_ans[i]=0;
		correct_ans[i]=0;
	}
	for( i=0;i<bpnn.test_length_array;++i){
		T *temp=compute_value(bpnn.input_test[i]);
		/*cout<<">预测结果："<<endl;
		print_output(temp);
		cout<<"实际结果："<<endl;
		print_output(bpnn.output_test[i]);
		*/

		int t1=find_index_of_max_number(temp,bpnn.output_lay_num);
		int t2=find_index_of_max_number(bpnn.output_test[i],bpnn.output_lay_num);
		if(t1==t2)
		    prediction_ans[t2]++;
		correct_ans[t2]++;
	}
	cout<<"正确率"<<endl;
	for(i=0;i<bpnn.output_lay_num;++i){
	    cout<<1.0*prediction_ans[i]/correct_ans[i]<<":"<<prediction_ans[i]<<"/"<<correct_ans[i]<<endl;
	}
}
void test_bpnn(){

	T *e =new T[bpnn.output_lay_num];

	for(int i=0;i<bpnn.test_length_array;++i){
		//预测结果：
		cout<<">预测结果："<<endl;
		T *temp=compute_value(bpnn.input_test[i]);
	    print_output(temp);
	    
		//实际结果：
		cout<<"实际结果："<<endl;
		print_output(bpnn.output_test[i]);
		for(int j=0;j<bpnn.output_lay_num;++j){
		    e[j]+=fabs(temp[j]-bpnn.output_test[i][j]);
		}
		delete temp;
	}
	for(int j=0;j<bpnn.output_lay_num;++j){
		e[j]/=bpnn.test_length_array;
	}
	cout<<"误差结果："<<endl;
	print_output(e);
	delete e;
}

int main(){
	//srand((unsigned)time(NULL));
	init();
	time_t t_start, t_end;
	t_start = time(NULL) ;
	bpnn_train();
	t_end = time(NULL) ;
	printf("time: %.0f s\n", difftime(t_end,t_start)) ;
	char a;
	cout<<"输入字符，测试神经网络正确率"<<endl;
	cin>>a;
	test_voice_bpnn();
	cin>>a;
    return 0;
}