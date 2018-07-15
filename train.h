#include <omp.h>
#include "mpi.h"

#define 	TRAIN_FILE_NAME  					"train.txt"
#define 	TEST_FILE_NAME  					"test.txt"
#define		U											116020
#define		V											136737
#define   Train_data_num						44302276																
#define   Test_data_num						1160200
#define		H											32
#define		alpha									0.0001
#define  	beta											0.02
#define		thread_num							8

typedef struct{
	unsigned int		uid;
	unsigned int 		vid;
	unsigned short	rate;	
} data;

double cpu1, cpu2;

void train_scatter(data *Train_data, float * user_FV, float * item_FV, float * user_Update, float * item_Update, int sid){
	float prediction;		
	float error;
	int i,j, uid, vid;
	if(sid==0){
		#pragma omp parallel for default(shared) private(i, j, vid, uid, prediction, error) schedule(static)
		for(i=0; i<Train_data_num/2; i++){
			prediction = 0;
			uid = Train_data[i].uid;
			vid = Train_data[i].vid;		
			for(j=0; j<H; j++){				
				prediction+= user_FV[uid*H+j]*item_FV[vid*H+j];
			}	
			error = Train_data[i].rate-prediction;		
			for(j=0; j<H; j++){
				user_Update[uid*H+j] = 2*alpha*error*item_FV[vid*H+j]-beta*user_FV[uid*H+j];
				item_Update[vid*H+j] = 2*alpha*error*user_FV[uid*H+j]-beta*item_FV[vid*H+j];	
			}
		}
	}
	else {
		#pragma omp parallel for default(shared) private(i, j, vid, uid, prediction, error) schedule(static)
		for(i=Train_data_num/2; i<Train_data_num; i++){
			prediction = 0;
			uid = Train_data[i].uid;
			vid = Train_data[i].vid;		
			for(j=0; j<H; j++){				
				prediction+= user_FV[uid*H+j]*item_FV[vid*H+j];
			}	
			error = Train_data[i].rate-prediction;		
			for(j=0; j<H; j++){
				user_Update[uid*H+j] = 2*alpha*error*item_FV[vid*H+j]-beta*user_FV[uid*H+j];
				item_Update[vid*H+j] = 2*alpha*error*user_FV[uid*H+j]-beta*item_FV[vid*H+j];	
			}
		}
	}
	//MPI_Barrier(MPI_COMM_WORLD);
} 

void train_gather(float * user_FV, float * item_FV, float * user_Update, float * item_Update, int sid){
	int i,j;	
		
	#pragma omp parallel for default(shared) private(i, j) schedule(static)
	for(i=U/2*sid; i<U/2*(sid+1); i++){
		for(j=0; j<H; j++){
			user_FV[i*H+j] = user_FV[i*H+j] + user_Update[i*H+j];	
			user_Update[i*H+j] = 0;		
		}
	}
	#pragma omp parallel for default(shared) private(i, j) schedule(static)
	for(i=0; i<V; i++){
		for(j=0; j<H; j++){
			item_FV[i*H+j] = item_FV[i*H+j] + item_Update[i*H+j];	
			item_Update[i*H+j] = 0;		
		}
	}
} 

float validate(data * Test_data, float * user_FV, float * item_FV, int sid){
		float Overall_squar_error = 0;
		float prediction;		
		int i,j;
		
		if(sid==0){
			#pragma omp parallel for default(shared) private(i, prediction) schedule(static) reduction(+:Overall_squar_error)
			for(i=0; i<580100; i++){
					prediction = 0;		
					for(j=0; j<H; j++){				
						prediction+= user_FV[Test_data[i].uid*H+j]*item_FV[Test_data[i].vid*H+j];
					}	
					Overall_squar_error += (Test_data[i].rate-prediction)*(Test_data[i].rate-prediction);
			}
		}
		if(sid==1){
			#pragma omp parallel for default(shared) private(i, prediction) schedule(static) reduction(+:Overall_squar_error)
			for(i=580101; i<Test_data_num; i++){
					prediction = 0;		
					for(j=0; j<H; j++){				
						prediction+= user_FV[Test_data[i].uid*H+j]*item_FV[Test_data[i].vid*H+j];
					}	
					Overall_squar_error += (Test_data[i].rate-prediction)*(Test_data[i].rate-prediction);
			}
		}
		MPI_Allreduce(&Overall_squar_error, &Overall_squar_error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);	
		MPI_Barrier(MPI_COMM_WORLD);
		return Overall_squar_error/Test_data_num;			
}	