#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "train.h"


int main(int argc, char **argv) {
	FILE *fp;
	int i, j, k, n;
	unsigned int u, v, r;
	unsigned int u_max = 0;
	unsigned int v_max = 0;
	int iteration;
	struct timespec start, stop; 
	double exe_time;
	float  MSE;
	int sid;
	
	MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &sid);  /* My processor ID */
  printf("My rank is %d\n", sid);
  		
	data 	*Train_data = (data*)malloc(sizeof(data)*Train_data_num);		
	data 	*Test_data 	= (data*)malloc(sizeof(data)*Test_data_num);
			
	fp=fopen(TRAIN_FILE_NAME,"r");
	for(i=0;i<Train_data_num;i++){
		if(fscanf(fp,"%d %d %d\n",&u,&v,&r)!=EOF){
			Train_data[i].uid=u;
			Train_data[i].vid=v;
			Train_data[i].rate=r;
			if(u>u_max) u_max = u;
			if(v>v_max) v_max = v;
		}
	}			
	fclose(fp);			
	
	fp=fopen(TEST_FILE_NAME,"r");
	for(i=0;i<Test_data_num;i++){
		if(fscanf(fp,"%d %d %d\n",&u,&v,&r)!=EOF){
			Test_data[i].uid=u;
			Test_data[i].vid=v;
			Test_data[i].rate=r;
			if(u>u_max) u_max = u;
			if(v>v_max) v_max = v;
		}
	}			
	fclose(fp);	
					
	printf("u_max = %d, v_max =%d\n", u_max, v_max);	
	float *user_FV = (float *)malloc(sizeof(float)*U*H);
	float *item_FV = (float *)malloc(sizeof(float)*V*H);
	float *user_Update = (float *)malloc(sizeof(float)*U*H);
	float *item_Update = (float *)malloc(sizeof(float)*V*H);
	float *item_Update_recvbuf = (float *)malloc(sizeof(float)*V*H);
	
	srand((unsigned int)time(NULL));	
	for(i=0; i<U*H; i++)  user_FV[i] 		=  	(float) rand() / (RAND_MAX);
	for(i=0; i<U*H; i++)  user_Update[i] =  	0;
	for(i=0; i<V*H; i++)  item_FV[i] 		= 	(float) rand() / (RAND_MAX);
	for(i=0; i<V*H; i++)  item_Update[i] = 	0;
	for(i=0; i<V*H; i++)  item_Update_recvbuf[i] = 	0;
	
	omp_set_num_threads(thread_num);	
	
	MPI_Barrier(MPI_COMM_WORLD); 
	cpu1 = MPI_Wtime();
	
	for(iteration =0; iteration<10; iteration++){
		if(clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");} 
		train_scatter(Train_data, user_FV, item_FV, user_Update, item_Update, sid);
		
		//MPI_Allreduce((float*)item_Update, (float*)item_Update_recvbuf, H*V, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);	
		
		//train_gather(user_FV, item_FV, user_Update, item_Update_recvbuf, sid);
		//MSE = validate(Test_data, user_FV, item_FV, sid);
		if(clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime");} 	
		exe_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		//if (sid == 0) printf("sid %d: Iter %d -> MSE = %.3f, exe time = %f\n", sid, iteration, MSE, exe_time);
		if (sid == 0) printf("%.3f\n", MSE);
	}
	
	cpu2 = MPI_Wtime();
	if (sid == 0) printf("Total time= %le\n",cpu2-cpu1);
		
	//if(clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime");} 	
	//exe_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;	
	//printf("Execution time is %f sec per iteration, n=%d\n", exe_time/iteration, iteration);
	
	////////free space////////
			
	free(user_FV);
	free(item_FV);	
	free(user_Update);	
	free(item_Update);	
	free(Train_data);
	free(Test_data);
	
	MPI_Finalize(); /* Clean up the MPI environment */
	return 0;
}


